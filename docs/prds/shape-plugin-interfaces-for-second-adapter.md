# PRD: Shape plugin interfaces for the second adapter

> **Status:** updated after design grill. Earlier draft proposed `ChunkRequest` and treated retrieval as input-only reshape; both have been revised below. See change log at the bottom.

## Problem Statement

This module is a plugin framework for RAG. The README is explicit: parser, chunker, embedder, vector DB, and retrieval are extension points, and the developer pitch is *"swap one stage without rewriting the rest."*

The retrieval plugin point is shaped against the only adapter that currently exists, not against the adapters the README explicitly anticipates. As a developer trying to add the *next* retrieval adapter, the seam doesn't fit:

- `BaseRetrievalStrategy.retrieve(embedded_query, top_k, filters)` is byte-identical to `vector_db.retrieve` and returns bare **Chunks** with no score. A re-ranker needs the raw query text *and* candidate scores. A hybrid (BM25 + dense) strategy needs the raw query text for the lexical leg. MMR needs scores to diversify. None of those can be added without reaching around the interface.

The chunker plugin point has a related but smaller problem: `BaseChunker.chunk(parsed_document, metadata)` accepts a typed `DocumentParserResult` plus a loosely-typed `metadata: dict` whose only documented expected key is `tags`, an unwritten convention specific to one chunker. A second chunker has to discover what's allowed in that dict from existing source code.

## Solution

Reshape the retrieval plugin point so that:

- Every retrieval strategy receives one **`RetrievalRequest`** carrying both the raw query text and the embedding (plus `top_k` and `filters`). Both `query_text` and `query_embedding` are required.
- Every retrieval strategy returns **`List[ScoredChunk]`**. **`ScoredChunk`** is a new wrapper type pairing a **Chunk** with the score it earned against a specific query.
- The vector DB always returns **`List[ScoredChunk]`** from its `retrieve` method, surfacing scores instead of discarding them.
- The `/retrieve` HTTP response wraps each result as `{chunk, score}`, mirroring `ScoredChunk` directly.
- Scores are **raw and strategy-specific**; the framework does not normalize across strategies. See [ADR-0001](../adr/0001-retrieval-scores-are-raw-and-strategy-specific.md).

Reshape the chunker plugin point trivially:

- `BaseChunker.chunk` and `DocumentChunker.chunk` rename the second parameter from `metadata` to `extra`. Add a docstring clarifying that `extra` is chunker-specific and the framework makes no claim about its keys. The chunker continues to own chunk metadata composition and chunk IDs (per the README's stated design rule).

## User Stories

1. As a framework contributor, I want to add a re-ranking retrieval strategy, so that I can implement reranker-based RAG without modifying `BaseRetrievalStrategy` or `RAGPipeline`.
2. As a framework contributor, I want to add a hybrid (BM25 + dense) retrieval strategy, so that I can fuse lexical and semantic results without bypassing the retrieval seam.
3. As a framework contributor, I want every retrieval strategy to receive the raw query text, so that strategies that need it (re-rankers, query expanders) don't have to plumb it through a side channel.
4. As a framework contributor, I want every retrieval strategy to receive candidate scores from the vector DB, so that re-rankers, hybrid strategies, and MMR can do their job without reaching around the interface.
5. As a framework contributor, I want the vector DB to surface the scores it already has, so that this information is not silently discarded between layers.
6. As an API consumer, I want to see the score of each retrieved chunk in the `/retrieve` response, so that I can apply confidence thresholds, deduplicate near-duplicates, or weight results in downstream prompting.
7. As an API consumer, I want score semantics documented per strategy rather than normalized across strategies, so that I retain the full information each strategy produces and can interpret scores correctly.
8. As a framework contributor, I want `ScoredChunk` to be a separate type rather than a field on `Chunk`, so that a `Chunk` read outside a query context (e.g. by id, after upsert) does not carry a meaningless score.
9. As a framework contributor, I want to add a section-aware chunker, so that I can chunk by parsed sections without re-deriving which caller-metadata keys are honored.
10. As a framework contributor, I want the chunker's loosely-typed input parameter to be named `extra` and documented as chunker-specific, so that I do not assume the framework guarantees any of its keys.
11. As a framework contributor, I want `tags` to remain a `DocumentChunker` convention rather than a framework-level field, so that other chunkers are not forced to adopt it.
12. As a framework contributor, I want the chunker to keep ownership of chunk metadata and chunk IDs, so that the framework's stated architectural rule remains true after this change.
13. As a maintainer, I want the dashboard to continue working without code changes, so that the public-facing UX is not gated on this PRD.
14. As a maintainer, I want the strategy discovery endpoint (`/strategies`) to continue returning the same shape, so that machine consumers of that endpoint are unaffected.
15. As a maintainer, I want existing pipeline integration tests to continue passing after migration (with assertions updated for the new return shape), so that I have confidence the end-to-end behavior is preserved.
16. As a maintainer, I want the migration to land without backward-compatibility shims for the old signatures, so that the codebase has one clear contract.
17. As a contributor writing tests for a retrieval strategy, I want to construct a `RetrievalRequest` directly, so that my tests do not need to mock `RAGPipeline` to exercise the strategy.
18. As a contributor reading the codebase, I want the retrieval strategy's input contract expressed as a Pydantic model, so that the schema is enforced and self-documenting.
19. As a contributor reading the codebase, I want `ScoredChunk` defined as a Pydantic model, so that the score's existence at every retrieval-aware seam is self-documenting.
20. As a maintainer, I want `Chunk`, `ChunkMetadata`, and the embedder/vector-DB upsert paths to stay unchanged in their indexing behavior, so that this PRD does not bleed into the chunker or embedder plugin points beyond the rename.

## Implementation Decisions

**Modules to be built or modified:**

- **New typed model: `ScoredChunk`** — Pydantic model with fields `chunk: Chunk` and `score: float`. Lives next to `BaseRetrievalStrategy` in the retrieval module.
- **New typed model: `RetrievalRequest`** — Pydantic model with fields `query_text: str` (required), `query_embedding: list[float]` (required), `top_k: int = 5`, `filters: dict | None = None`. Lives next to `BaseRetrievalStrategy`.
- **`BaseVectorMemory.retrieve(...)`** — return type changes from `List[Chunk]` to `List[ScoredChunk]`. Signature otherwise unchanged.
- **`QdrantVectorDB.retrieve(...)`** — preserves Qdrant's similarity score (currently discarded) and returns `List[ScoredChunk]`.
- **`BaseRetrievalStrategy.retrieve(request)`** — signature becomes one `RetrievalRequest` argument; return type becomes `List[ScoredChunk]`.
- **`SimilarityRetrievalStrategy.retrieve(request)`** — migrates to the new interface; ignores `query_text`; passes embedding/top_k/filters through to the vector DB.
- **`RAGPipeline.retrieve`** — assembles a `RetrievalRequest` from the existing inputs (HTTP/CLI query string, embedded query, top_k, filters) and returns `List[ScoredChunk]`.
- **HTTP `/retrieve` response** — each result is serialized as `{chunk: {...}, score: 0.87}`, mirroring `ScoredChunk` directly via Pydantic.
- **`BaseChunker.chunk` and `DocumentChunker.chunk`** — second parameter renamed `metadata` → `extra`. Docstring added clarifying that `extra` is chunker-specific.
- **`example_usage.py`** — updated to consume the new `vector_db.retrieve` return shape (read `.chunk` from each `ScoredChunk`).

**Architectural decisions:**

- The chunker continues to own chunk metadata composition and chunk IDs (this PRD does not extract a separate metadata-assembly module; that contradicts the README's stated rule).
- `tags` is *not* a framework concept. It is a `DocumentChunker` convention and stays that way. No typed slot for `tags` on any framework model.
- `ChunkRequest` is **not** introduced. After removing `tags` from the framework's contract, wrapping `(parsed_document, extra)` in a Pydantic model would be repackaging without depth.
- `ScoredChunk` is a separate type from `Chunk`. Score is meaningful only against a query and does not belong on the chunk's identity. See [CONTEXT.md](../../CONTEXT.md).
- Scores are raw and strategy-specific, never normalized. See [ADR-0001](../adr/0001-retrieval-scores-are-raw-and-strategy-specific.md).
- `vector_db.retrieve` always returns `List[ScoredChunk]`. No second method, no opt-in mode. Today's only callers (`SimilarityRetrievalStrategy` and `example_usage.py`) migrate.
- `RetrievalRequest.query_text` and `query_embedding` are both required. No optional/`None` paths.
- HTTP response shape mirrors `ScoredChunk` (wrapped, not flat). The internal model and external contract stay in sync.
- No backward-compatibility shims for the old signatures or response shape. Migration is atomic per slice.
- `/strategies` endpoint, dashboard, embedder, and indexing path are unchanged.

## Testing Decisions

**What makes a good test for this PRD:** tests should exercise the *external behavior* of the seams being reshaped — the input the strategy/chunker receives, the output it produces, and the HTTP response shape — and should not assert on private helpers. The new typed models (`RetrievalRequest`, `ScoredChunk`) do not need direct tests; Pydantic enforces the schema and value comes from how callers/adapters use the model.

**Modules to be tested:**

1. **`SimilarityRetrievalStrategy` against `RetrievalRequest`** — construct a `RetrievalRequest`, hand it to the strategy with a stub vector DB, assert the vector DB receives the right embedding/top_k/filters, the strategy returns `List[ScoredChunk]`, and `query_text` is correctly ignored. Prior art: `tests/retrieval/test_similarity_retrieval.py`.
2. **`QdrantVectorDB.retrieve` returns `ScoredChunk`** — assert scores from the Qdrant client are surfaced (not discarded). Prior art: existing vector DB tests under `tests/vector_db/`.
3. **`DocumentChunker` after rename** — assert it still produces correct chunks and metadata; the rename is mechanical, but a smoke test confirms no behavior drift. Prior art: existing chunker tests under `tests/chunking/`.
4. **`RAGPipeline.retrieve` integration** — with a stub strategy, assert the pipeline builds a `RetrievalRequest` carrying the raw query text and embedding, and returns `List[ScoredChunk]` end-to-end. Prior art: existing pipeline tests under `tests/pipeline/`.
5. **`/retrieve` HTTP response shape** — assert the response wraps each result as `{chunk, score}`. Prior art: existing API tests under `tests/app/`.

`RetrievalRequest` and `ScoredChunk` Pydantic models are not tested directly.

## Out of Scope

- Wiring or deleting the orphan `Storage` module.
- Consolidating the five factory files.
- Moving `FileOps.validate` logic into individual parsers.
- Removing the duplicate `_strategy`/`_backend` aliases in `retrieval_factory`.
- Embedder strategy changes.
- Adding the second retrieval adapter (re-ranker, hybrid, MMR). This PRD only reshapes the seam to make those additions possible; the adapters themselves are separate work.
- Adding a second chunker.
- Backward-compatibility shims.
- Dashboard UX changes to display the new `score` field. The dashboard pretty-prints the JSON response and continues to work; *displaying* scores is a separate UX decision.
- Score normalization across strategies. Recorded in [ADR-0001](../adr/0001-retrieval-scores-are-raw-and-strategy-specific.md).
- A `score_kind` discriminator on `ScoredChunk`. Considered and deferred; can be added additively later.
- Optional `query_text`. Strict-by-default; loosen later only if a real use case appears.

## Further Notes

- The README's developer mindset section (*"chunkers own chunk formation and chunk metadata"*) is load-bearing for this PRD's design. An earlier draft of this work proposed extracting metadata composition out of the chunker; that draft was withdrawn after the README was re-read.
- An earlier draft of this PRD also proposed introducing `ChunkRequest` as a typed wrapper around `(parsed_document, metadata)`. The grill revealed that, once `tags` is acknowledged as a `DocumentChunker` convention rather than a framework concept, `ChunkRequest` would be a Pydantic wrapper around the same two arguments — repackaging, not depth. It was dropped.
- After this PRD lands, the natural follow-up is to actually add a second retrieval adapter (e.g. `RerankRetrievalStrategy`). That will validate whether the reshaped interface is *deep* in practice — the deletion test for this PRD is *"does the second adapter slot in cleanly without re-shaping the interface again?"*

---

## Change log

- **Initial draft.** Two slices proposed: widen `BaseRetrievalStrategy` input + introduce `ChunkRequest`. Both treated as input-only reshapes. HTTP contract assumed unchanged.
- **After design grill.** Retrieval reshape extended to scored output (vector DB + strategy + pipeline + HTTP all return `ScoredChunk`). HTTP response shape changes to wrapped `{chunk, score}`. `ChunkRequest` dropped after Q9–Q10 revealed it as repackaging. Chunker work collapses to a parameter rename + docstring. ADR-0001 records the raw-scores decision. CONTEXT.md adds `Chunk` and `ScoredChunk`.
