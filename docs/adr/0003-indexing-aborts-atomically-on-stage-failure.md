# Indexing aborts atomically on stage failure

`RAGPipeline.indexer` processes a document through parser → chunker → embedder → vector DB. When any stage call raises — most commonly an embedder API error mid-loop or a vector DB write failure — the pipeline aborts the entire indexing job and surfaces the failure to the caller. No chunks from a partially-failed job are persisted; the corpus state after a failed index is identical to the corpus state before.

The pipeline does **not** offer a "skip the bad chunk and continue" mode.

## Why

- **Partial indexing produces silently inconsistent corpora.** If chunk 7 of 100 fails to embed and the pipeline continues, the corpus afterward contains 99 chunks of a document that the operator believes was indexed completely. A later retrieval against that document returns plausible-looking but incomplete results, and there is no signal — to the LLM, to the operator, or to the agent harness — that anything went wrong. This is the worst failure mode in a RAG system: confident wrong answers downstream of silent partial state.
- **The "skip and continue" temptation is real.** A future contributor will hit a flaky embedder API, see the indexer abort, and want to add `try/except: continue` inside the embedding loop ([rag_pipeline.py:115-119](../../memory_module/rag_pipeline.py#L115)). That change looks small and helpful in a PR diff. It is neither. This ADR exists to gatekeep that PR.
- **Atomic abort matches the agent-tool contract.** The boundary in PRD-2 returns one outcome per call: success or `"Error: ..."`. There is no shape for "succeeded with N of M chunks indexed." Until that shape exists and a consumer demonstrably needs it, atomic abort is the only outcome the contract supports honestly.
- **Reversible additively.** If a real consumer appears that genuinely benefits from partial indexing (e.g. a long-running batch index where one bad chunk in 10,000 should not block the rest), the path forward is an explicit `partial=True` parameter with a documented result shape that surfaces *which* chunks failed. That is purely additive; the default stays atomic.

## Considered alternatives

- **Skip-and-continue by default.** Rejected for the reasons above. The silent-partial-state failure mode is severe enough that it cannot be the default.
- **Skip-and-continue with a warning log.** Rejected. A log line is invisible to the LLM and to most operators; it does not change the inconsistency, only the audit trail of it.
- **Atomic abort plus an explicit `partial=True` mode shipped now.** Rejected as speculative. No consumer needs it today; ship the default, add the mode when a real use case justifies the result-shape design.
