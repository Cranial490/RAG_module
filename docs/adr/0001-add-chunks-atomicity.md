# ADR 0001: `add_chunks` atomicity model

- Status: Accepted
- Date: 2026-06-30

## Context

`RAGPipeline.indexer` calls `vector_db.add_chunks(chunks)` as its final stage.
The current backing implementation, `QdrantVectorMemory.add_chunks`
(`memory_module/vector_db/qdrant_vector_db.py`), issues one batched
`client.upsert(points=[...], wait=True)` call.

Qdrant's `upsert` is **not transactional across a batch**. If the call fails
partway (network drop, server-side error mid-batch), some points in the batch
may already be persisted while others are not. The indexer then re-raises the
underlying failure as `VectorDBFailed`, but the caller has no signal about
whether the collection now contains a partial write.

The Slice 2 commit (`03c923e`, "Slice 2: stage-failure errors ... + atomic
abort") uses the phrase "atomic abort". That phrase refers only to
**pre-`add_chunks` atomicity** — an embedder failure ends the indexer job
before `add_chunks` is called, so the vector DB is untouched. It does **not**
claim that `add_chunks` itself is atomic. This ADR exists partly to reconcile
that ambiguous phrasing with the actual contract.

## Decision

`add_chunks` provides a **best-effort compensating delete** guarantee.

On any exception raised by the underlying `upsert`, the implementation issues a
`delete` for every `chunk_id` in the batch and then re-raises the original
failure wrapped as `RuntimeError` (which the pipeline layer wraps as
`VectorDBFailed`). The compensating delete is itself best-effort: if it also
raises, its exception is swallowed and the original upsert error is still what
surfaces.

Concretely, the contract callers can rely on:

- **Success path**: the batch is persisted; no delete is issued.
- **Failure path**: `VectorDBFailed` (or, at the vector-db-layer boundary,
  `RuntimeError("Failed to add chunks to Qdrant: ...")`) is raised, and the
  implementation has made a best-effort attempt to remove any points from the
  failed batch. In the rare case both the upsert and the compensating delete
  fail, orphan points may remain in the collection. Callers that need a
  stronger guarantee should dedupe by `chunk_id` on retry.

## Considered alternatives

1. **Best-effort, no cleanup.** Accept partial state on failure; document it.
   Zero code cost, weakest guarantee. Rejected because the extra round-trip on
   the failure path is cheap and materially reduces orphan probability.
2. **Per-point upsert with rollback tracking.** Upsert points one-by-one, track
   successes, delete succeeded IDs on failure. Rejected: pays a per-point
   network round-trip on the success path (N× latency for a batch of N
   chunks), and still doesn't provide true ACID — a process crash between
   "upsert of point k succeeded" and "record k in success list" still loses
   track. The extra guarantee didn't justify the latency cost.

## Consequences

- The pipeline continues to raise `VectorDBFailed` when `add_chunks` fails.
  Callers should treat that as "the batch was rejected; the DB has been
  best-effort restored to its pre-call state, but you should still dedupe by
  `chunk_id` on retry."
- The "atomic abort" phrasing in commit `03c923e` should be read as
  "pre-`add_chunks` atomicity only". This ADR is the authoritative source for
  the `add_chunks` contract.
- Future vector-db backends implementing `BaseVectorMemory.add_chunks` should
  honor this same contract: on failure, attempt to remove the batch's IDs
  before re-raising.
