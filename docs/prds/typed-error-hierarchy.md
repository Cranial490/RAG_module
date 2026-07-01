# PRD: Typed error hierarchy in core

> **Status:** rewritten after design grill (see change log). Originally bundled with the agent-tool boundary; split into this PRD (core hierarchy) + a follow-up PRD on the agent-tool boundary.

## Problem Statement

This module's failure surface today is a mix of `RuntimeError`, `ValueError`, and `TypeError` raised across [rag_pipeline.py:19](../../memory_module/rag_pipeline.py#L19), [rag_pipeline.py:95-110](../../memory_module/rag_pipeline.py#L95), and [rag_pipeline.py:130-135](../../memory_module/rag_pipeline.py#L130). The exception *type* carries no information about which plugin stage failed or what class of failure occurred. Any caller that wants to distinguish "parser rejected this file" from "embedder API call timed out" from "the operator gave us a malformed pipeline config" has to string-match prose.

This is a problem for every current and anticipated consumer:

- The FastAPI app in [main.py](../../main.py) wraps every internal raise as `HTTPException(400)` with the original prose embedded — no machine-readable code, no way for a UI to render different errors differently.
- An agent-harness adapter (planned, see follow-up PRD) needs discrete error codes so the LLM and the harness's logs can distinguish failure classes.
- `RAGPipeline._resolve_component` already string-matches its own error messages ([rag_pipeline.py:84-92](../../memory_module/rag_pipeline.py#L84)) to detect "invalid key" vs. "constructor crashed" — direct evidence that the codebase is already paying the cost of untyped failures.

The CONTEXT.md glossary names plugin stages (parser, chunker, embedder, vector DB, retrieval) as the module's core architectural vocabulary. The failure surface should align with that vocabulary — not symmetrically, but where the alignment carries real domain meaning. See [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).

## Solution

Introduce a typed exception hierarchy in `memory_module/errors.py` and convert the relevant raise sites in `RAGPipeline` to use it. The hierarchy is consumer-agnostic: HTTP, CLI, MCP, and a future agent-tool boundary all consume the same exceptions, each translating to whatever shape its caller wants.

**Hierarchy** (flat, all inheriting from `RAGError`):

| Class | Raised when |
|---|---|
| `ConfigError` | `RAGPipeline.__init__` receives bad config; factory rejects an unknown strategy key or fails to construct a strategy |
| `InvalidQuery` | `RAGPipeline.retrieve` receives an empty/whitespace query |
| `ParserRejected` | `parser.accepts(document)` returns False (pre-flight refusal, see CONTEXT.md → Parser Rejection) |
| `EmbedderFailed` | The embedder stage call raises any exception during `indexer` or `retrieve` |
| `VectorDBFailed` | The vector DB stage call raises any exception during `indexer` or `retrieve` |
| `NoChunksProduced` | `RAGPipeline.indexer` produces zero chunks after the chunker stage (root cause may be parser or chunker) |

**Out of the hierarchy by design:** failures inside the chunker or retrieval stages get no named class — they propagate as whatever the plugin raises. See [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).

**Placement of the `try/except` wraps:** `RAGPipeline.indexer` and `RAGPipeline.retrieve` wrap each external stage call (embedder, vector DB) in its own `try/except Exception` and re-raise as the typed exception via `raise EmbedderFailed(...) from e`. Plugin implementations themselves do not import the error hierarchy. Rationale: the pipeline is already the stage classifier (`_resolve_component` does this for factories); plugin authors don't need to know about the framework's error taxonomy.

**Indexing behavior on stage failure:** atomic abort. The first embedder or vector DB failure ends the index job; no chunks from a partially-failed job are persisted. See [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md).

## User Stories

1. As an HTTP API consumer, I want different RAG failures to be distinguishable by type (not just by prose), so that my UI / client code can render `ParserRejected` differently from `EmbedderFailed` without parsing strings.
2. As a future agent-tool adapter (separate PRD), I want core RAG failures to come up as typed exceptions, so that I can map each one to a stable `error_code` at my boundary without re-classifying prose.
3. As a framework contributor adding a new retrieval strategy or chunker, I want to *not* need to import any RAG error class, so that my plugin contract stays minimal and the pipeline does the classification.
4. As a framework contributor adding a new embedder or vector DB adapter, I want my exceptions to propagate naturally, so that I do not write `raise EmbedderFailed(...)` myself — `RAGPipeline` wraps me.
5. As a framework contributor adding a new parser, I want to understand that `accepts()` returning False is a **Parser Rejection** (distinct from a parse-time crash), so that I implement the pre-flight check correctly. (See CONTEXT.md.)
6. As a maintainer, I want a single file (`memory_module/errors.py`) where the entire failure taxonomy is discoverable, so that "what can this module raise?" has one answer.
7. As a maintainer, I want the existing FastAPI app to continue working unchanged in this PRD, so that the human-facing UX is not blocked. Mapping `RAGError` subclasses to HTTP status codes is a separate, additive follow-up.
8. As a maintainer, I want `_resolve_component`'s prose-matching ([rag_pipeline.py:84-92](../../memory_module/rag_pipeline.py#L84)) to be unblocked by this PRD, so that the follow-up cleanup (factories raise typed errors, resolver dispatches on type) becomes a small mechanical change. This PRD does **not** do that cleanup — it makes it possible.
9. As an operator, I want indexing to abort atomically on the first embedder or vector DB failure, so that the corpus never ends up in a silently inconsistent state (99 of 100 chunks indexed, no signal). See [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md).
10. As a contributor writing tests for `RAGPipeline`, I want to assert on exception *type* rather than message content, so that test assertions are stable against prose changes.
11. As a contributor reading the codebase, I want `ParserRejected` to exist as a separate class from any general parser failure, so that the pre-flight refusal concept (CONTEXT.md → Parser Rejection) is reflected in the code, not just the glossary.
12. As a contributor reading the codebase, I want to see *why* chunker and retrieval don't have named exceptions, so that I do not perceive the asymmetry as an oversight. (Answered by [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).)
13. As a maintainer, I want this PRD to land without backward-compatibility shims for the old `RuntimeError`/`ValueError`/`TypeError` signatures, so that the codebase has one clear failure contract.

## Implementation Decisions

**Modules to be built or modified:**

- **New module: `memory_module/errors.py`** — defines `RAGError` (base, with `code: str` class attribute) and the six subclasses listed above. Subclass `code` values: `config_error`, `invalid_query`, `parser_rejected`, `embedder_failed`, `vector_db_failed`, `no_chunks_produced`. (Codes live on the class so PRD-2's boundary can read them without a separate mapping table.)
- **`memory_module/rag_pipeline.py`** — converts the existing raise sites:
  - `__init__` `TypeError("RAGPipeline config must be a dict.")` → `ConfigError`
  - `_resolve_component` `ValueError` paths → `ConfigError`
  - `indexer` parser-rejection branch → `ParserRejected` (replaces today's `ValueError("Parser does not accept...")`)
  - `indexer` missing-stage `RuntimeError`s → `ConfigError`
  - `retrieve` missing-stage `RuntimeError`s → `ConfigError`
  - `retrieve` empty-query `ValueError` → `InvalidQuery`
- **`memory_module/rag_pipeline.py` — new `try/except` wraps** around stage calls in `indexer` and `retrieve`:
  - `indexer`: one wrap around `embedder.embed(...)` (whole loop) raising `EmbedderFailed`; one wrap around `vector_db.add_chunks(...)` raising `VectorDBFailed`.
  - `retrieve`: one wrap around `embedder.embed(query)` raising `EmbedderFailed`; one wrap around `retriever.retrieve(...)` raising `VectorDBFailed` *only when the underlying cause is a vector DB call* — to be revisited if retrieval strategies start doing more than calling the vector DB. (For today's only strategy, this is correct.)
- **`memory_module/rag_pipeline.py` — new `NoChunksProduced` check** in `indexer`, after `chunker.chunk(...)` returns, before the embedding loop.
- **Existing plugin implementations** (parser, chunker, embedder, vector_db, retrieval) — **not modified**. They raise whatever they raise; the pipeline wraps.
- **`main.py`** (FastAPI app) — **not modified in this PRD**. It continues to catch generic `Exception` and raise `HTTPException(400)` with prose. A follow-up can map `RAGError` subclasses to HTTP status codes additively.

**Architectural decisions:**

- The hierarchy is **flat**. All six subclasses inherit `RAGError` directly. No intermediate categories (no `CallerError` / `ExecutionError`). Flat keeps the contract small; categorization can be added later if a consumer demonstrably needs to dispatch on a category.
- The pipeline is the **stage classifier**. Plugins remain unaware of the error hierarchy. This extends the existing pattern in `_resolve_component`. See [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).
- Only stages with **distinct domain semantics** get named exceptions. Chunker and retrieval failures propagate as-is. See [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).
- **`NoChunksProduced` is unified** across "parser produced empty content" and "chunker produced empty chunks". The post-chunk check covers both root causes; the exception's message can name which stage produced the empty output, but the *type* is one.
- **Atomic indexing abort.** First stage failure ends the job. See [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md).
- **No backward-compatibility shims.** Sites that raised `RuntimeError`/`ValueError`/`TypeError` for known-class failures are converted atomically. Callers that previously caught those types must update.
- **CONTEXT.md** is updated to add **Parser Rejection** as a domain term — the only glossary change this PRD makes.

## Testing Decisions

**What makes a good test for this PRD:** tests should assert on exception *type* and on the structural behavior (atomic abort, post-chunk check), not on exception message prose. The hierarchy's value is *types*, not strings; tests should reinforce that.

**Modules to be tested:**

1. **`RAGPipeline.__init__` raises `ConfigError`** — bad top-level config, unknown strategy key, factory constructor crash. Three parametrized cases. Replaces existing `TypeError`/`ValueError` assertions.
2. **`RAGPipeline.retrieve` raises `InvalidQuery`** on empty/whitespace query. Replaces existing `ValueError` assertion.
3. **`RAGPipeline.indexer` raises `ParserRejected`** when `parser.accepts(...)` returns False. Replaces existing `ValueError` assertion.
4. **`RAGPipeline.indexer` raises `NoChunksProduced`** when (a) the parser returns content but the chunker emits `[]`, and (b) the parser returns empty content (chunker naturally emits `[]`). Both produce the same exception type; assert that.
5. **`RAGPipeline.indexer` raises `EmbedderFailed`** when the stub embedder raises a generic exception mid-loop. Assert: (a) `EmbedderFailed` is raised, (b) `__cause__` is the original exception, (c) `vector_db.add_chunks` was **not** called (atomic abort, per [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md)).
6. **`RAGPipeline.indexer` raises `VectorDBFailed`** when the stub vector DB raises during `add_chunks`. Assert type and `__cause__`.
7. **`RAGPipeline.retrieve` raises `EmbedderFailed`** when the embedder fails on the query.
8. **`RAGPipeline.retrieve` raises `VectorDBFailed`** when the underlying retrieval strategy's vector DB call fails.
9. **Hierarchy invariant test** — assert every defined subclass inherits `RAGError` and has a non-empty `code` class attribute. (One test; catches accidental new classes that forget the `code` convention.)

Tests use stub plugins that raise on demand. No real parser / embedder / vector DB calls.

## Out of Scope

- The agent-tool boundary (`memory_module/agent_tool.py`), string-conversion to `"Error: ..."`, cancellation primitive, MCP wrapping. These are the follow-up PRD.
- Modifying `main.py` to map `RAGError` subclasses to HTTP status codes. Additive follow-up; not a blocker.
- Cleaning up `_resolve_component`'s prose-matching to dispatch on type instead. Unblocked by this PRD; not done here.
- Adding named exceptions for chunker or retrieval stage failures. Deliberately out by [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md).
- Introducing an `ErrorCode` enum (vs. the string `code` class attribute). Strings are sufficient; enum can be added later if a consumer benefits.
- Categorizing the hierarchy (caller-error vs. execution-error intermediate classes). Flat for now.
- Partial indexing / "skip and continue" mode. Deliberately out by [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md).
- Plugin authors raising typed exceptions themselves. The pipeline does the classification.
- Backward-compatibility shims for the old exception types.
- Updating CONTEXT.md beyond the **Parser Rejection** entry already added.

## Further Notes

- This PRD is a domain-shaped change, not an integration-shaped one. The hierarchy serves any consumer — HTTP, CLI, agent-tool, future MCP server — equally. The reason `main.py` is untouched here is not "HTTP doesn't deserve the hierarchy"; it's that mapping types to HTTP status codes is additive work that doesn't gate this PRD.
- The deletion test for this PRD: *"can the follow-up agent-tool boundary PRD be written without re-debating which failures get names?"* If yes, the hierarchy is the right shape. If the follow-up PRD wants to add (e.g.) `ChunkerFailed` because it has a real LLM-facing reason to distinguish it, that is a real signal — re-open [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md) rather than working around it.
- The natural follow-up after this PRD lands is the agent-tool boundary PRD (`rag_query` / `rag_index` signatures, cancellation contract, returned string format, MCP wrapping). That PRD consumes `RAGError` directly and adds nothing to the core's failure surface.

---

## Change log

- **Initial draft.** Bundled the typed exception hierarchy (core concern) with a string-conversion boundary mirroring the agent harness's `tool_executor` (integration concern). Proposed `memory_module/agent_tool.py` with `_await_with_cancel`, `RAGCancelled`, and the `"Error: <code>: ..."` return format.
- **After grill (Q1).** Split into two PRDs. This PRD covers the core hierarchy only; the boundary becomes a follow-up.
- **After grill (Q2).** Locked Option B taxonomy: only plugin stages with distinct domain semantics get named exceptions. Recorded as [ADR-0002](../adr/0002-error-taxonomy-mirrors-plugin-stages-selectively.md). Renamed `EmbeddingFailed` → `EmbedderFailed` for stage-name symmetry.
- **After grill (Q3).** Unified `NoChunksProduced` (parser-empty and chunker-empty produce the same exception). Check lives in `RAGPipeline.indexer`, post-chunk.
- **After grill (Q4).** Pipeline wraps stage calls; plugins remain unaware of the error hierarchy. Indexing aborts atomically on first stage failure; recorded as [ADR-0003](../adr/0003-indexing-aborts-atomically-on-stage-failure.md). Per-stage `try/except` (not one outer wrap) so classification stays distinct.
- **After grill (Q5).** Added **Parser Rejection** to CONTEXT.md. No other glossary changes.
- **After grill (Q6).** ADR-0002 and ADR-0003 written. Pipeline-wraps placement and flat hierarchy stay in the PRD body — not ADR-worthy.
