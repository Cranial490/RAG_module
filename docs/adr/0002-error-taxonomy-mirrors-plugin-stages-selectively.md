# Error taxonomy mirrors plugin stages selectively, not symmetrically

The typed exception hierarchy in `memory_module/errors.py` names a failure class only for plugin stages whose failures have **distinct domain semantics**. Today that means `ParserRejected`, `EmbedderFailed`, and `VectorDBFailed` are named; the chunker and retrieval stages are deliberately **not** named, even though they are plugin stages of equal architectural standing. Stage failures without a named class fall to the catch-all at the agent-tool boundary.

The hierarchy is therefore intentionally asymmetric with respect to the plugin-stage list in CONTEXT.md.

## Why

- **Honesty over symmetry.** `ParserRejected` exists because parsers have a pre-flight `accepts()` step (see CONTEXT.md → Parser Rejection) that produces a structurally different failure than "the parser crashed mid-conversion." `EmbedderFailed` and `VectorDBFailed` exist because those stages call external services, and "external service call failed" is a meaningful failure shape for retry / circuit-breaker logic. The chunker and retrieval stages have no equivalent distinct failure shape today — naming them would communicate a distinction that does not exist.
- **Generalizes [ADR-0001](0001-retrieval-scores-are-raw-and-strategy-specific.md).** That ADR rejected a `score_kind` discriminator as speculative contract without a real consumer. Same reasoning here: a symmetric `ChunkerFailed` / `RetrievalFailed` would be contract added on the assumption a future consumer needs to dispatch on chunker-vs-retrieval root cause. No such consumer exists; the cost (every plugin author internalizes another class; every boundary handler exhaustively dispatches) is paid today, the benefit is speculative.
- **Additive escape hatch.** If a real consumer appears that needs to branch on (say) chunker-specific failures, adding `ChunkerFailed` later is straightforward — new class, one new wrap site in `RAGPipeline.indexer`, callers opt-in. Removing a class once added is the harder direction; that asymmetry argues for "add later when justified."

## Considered alternatives

- **One named exception per plugin stage (symmetric).** Rejected. Pays the cost of contract breadth today for benefit that may never materialize. A future contributor reading the hierarchy would correctly infer "every stage has a named failure class" — but that inference is a fiction the framework would have to maintain forever, even for stages where the distinction is meaningless.
- **No named stage failures at all; one `RAGError` with a string `code`.** Rejected. `ParserRejected` carries a real domain distinction (pre-flight refusal vs. mid-flight crash) that deserves a type, not just a string. Collapsing it loses information the boundary and consumers will want.
