# Architecture Review — RAG_module

This module is a **plugin framework for RAG**, not application code. The README is explicit: parser / chunker / embedder / vector-DB / retrieval are documented extension points, and the developer pitch is *"swap one stage without rewriting the rest."* The seams are the product.

That reframes architecture review. The right question is **not** "does each seam earn its keep against today's callers?" — by that test, every plugin point with one adapter looks shallow. The right question is:

> Is each seam's interface shaped well enough to host the **second adapter** the README promises?

Vocabulary follows the skill glossary — *module*, *interface*, *seam*, *adapter*, *depth*.

---

## Current Architecture

### Top-level layout

```
RAG_module/
├── main.py                 FastAPI app — /index, /retrieve, /strategies endpoints
├── pipeline_main.py        CLI entry point
├── dashboard/              Static UI calling /index and /retrieve
├── memory_module/
│   ├── rag_pipeline.py     RAGPipeline — orchestrates parse → chunk → embed → store / retrieve
│   ├── parser/             DocxParser, base_parser, DocumentParserResult, FileMetadata
│   ├── chunking/           BaseChunker, DocumentChunker — owns chunk metadata + chunk IDs (intentional)
│   ├── embedder/           BaseEmbedder + concrete adapters
│   ├── vector_db/          BaseVectorMemory + concrete adapters (.retrieve, .upsert, ...)
│   ├── retrieval/          BaseRetrievalStrategy, SimilarityRetrievalStrategy
│   ├── storage/            Storage ABC, LocalStorage  ← never wired
│   ├── factory/            parser_factory, chunker_factory, embedder_factory,
│   │                       vector_db_factory, retrieval_factory  (one file per plugin point)
│   └── utils/file_operations.py    FileOps.validate(formats=[...])
└── tests/
```

### Indexing flow

```
HTTP /index  ──▶  RAGPipeline.index
                     │
                     ├─ parser_factory.get(...)        ──▶  DocxParser.parse(path)
                     │                                          └─ FileOps.validate(["docx"])
                     │
                     ├─ chunker_factory.get(...)       ──▶  DocumentChunker.chunk(parser_result, caller_meta)
                     │                                          ├─ reads parser_result.file_metadata
                     │                                          ├─ reads caller_meta dict ({"tags": ...})
                     │                                          └─ emits chunks + chunk-metadata (intentional ownership)
                     │
                     ├─ embedder_factory.get(...)      ──▶  embedder.embed(chunks)
                     │
                     └─ vector_db_factory.get(...)     ──▶  vector_db.upsert(embedded_chunks)

Storage module:  not called by anything.
```

### Retrieval flow

```
HTTP /retrieve  ──▶  RAGPipeline.retrieve
                       │
                       ├─ embedder.embed(query)
                       │
                       └─ retrieval_factory.get(...)
                              │
                              └─ SimilarityRetrievalStrategy.retrieve(eq, top_k, filters)
                                    │
                                    └─ vector_db.retrieve(eq, top_k, filters)   ← signature is identical
```

---

## Deepening Opportunities

### 1. Widen the retrieval interface so the next adapter actually fits

**Files:** [memory_module/retrieval/base_retrieval.py](memory_module/retrieval/base_retrieval.py), [memory_module/retrieval/similarity_retrieval.py](memory_module/retrieval/similarity_retrieval.py).

**Reframe:** The README names retrieval as a plugin point and lists hybrid retrieval, reranking, and custom filtering as the intended next adapters. So the seam is *not* the friction — its **shape** is. Today `BaseRetrievalStrategy.retrieve(embedded_query, top_k, filters)` is byte-identical to `vector_db.retrieve`, which means a re-ranker or hybrid adapter would have to reach around the interface to get what it needs:

- **Re-ranker** needs the raw query string (not just the embedding) and probably the chunk text + scores back from the vector DB.
- **Hybrid (BM25 + dense)** needs the raw query string for the lexical leg.
- **MMR / diversity** needs candidate scores and embeddings, not just the top-k chunks.

So the deepening is: shape the retrieval interface around *what retrieval strategies actually need*, not around what the vector DB happens to expose.

**Sketch (to grill, not commit):**

```python
class RetrievalRequest(BaseModel):
    query_text: str
    query_embedding: list[float]
    top_k: int
    filters: dict | None

class BaseRetrievalStrategy(ABC):
    def retrieve(self, request: RetrievalRequest) -> list[Chunk]: ...
```

`SimilarityRetrievalStrategy` ignores `query_text`; a re-ranker uses it. The vector DB adapter still owns the dense lookup; retrieval strategies compose on top.

**Benefit:** The seam stops being a pass-through *by accident*. Two-adapter test becomes reachable — adding `RerankRetrievalStrategy` requires zero changes to the interface.

---

### 2. Decide Storage's fate — wire it or delete it

**Files:** [memory_module/storage/storage.py](memory_module/storage/storage.py), [memory_module/storage/local_storage.py](memory_module/storage/local_storage.py), tests under [tests/storage/](tests/storage/).

**Problem:** `Storage` defines `save / read / delete` for raw documents, but nothing in `main.py`, `pipeline_main.py`, `RAGPipeline`, or the dashboard ever calls it. The README — which is otherwise meticulous about plugin points — does not mention it at all. One adapter, zero callers, zero documentation. This is the only candidate where the framework intent is genuinely silent.

**Two coherent answers, pick one:**

- **Wire it.** Make `Storage` the durable seam between "uploaded bytes" and "parser input." `/index` writes the upload through `Storage`, the parser reads from it. That gives you S3/GCS adapters, replay, and a real audit point.
- **Delete it.** No tests fail outside `tests/storage/`, no flow breaks, and the README pitch loses nothing.

**Benefit:** Either choice removes ambiguity. The current state — a documented-by-existence plugin point with no integration — is the worst of both.

---

### 3. Trim the noise inside the factory layer (don't consolidate it)

**Files:** [memory_module/factory/](memory_module/factory/) — five files, one per plugin point.

**Reframe:** I previously suggested merging the five factories into a single registry. Re-reading the README, "one file per plugin point" is itself a navigational aid that mirrors the conceptual layering ("Factories" is a documented architecture layer). Consolidating would optimize for line-count at the cost of the framework's own mental model. **Withdraw the merge.**

What's still defensible is narrower: [memory_module/factory/retrieval_factory.py](memory_module/factory/retrieval_factory.py) ships duplicate `register_retrieval_strategy` / `register_retrieval_backend` (and matching `get_*`) pointing at the same dict. Two names for one thing is unambiguous noise — pick one term and delete the alias.

**Benefit:** Locality of vocabulary. New contributors see one verb per plugin point and don't have to wonder which is canonical.

---

### 4. Normalize the chunker's input contract — keep ownership where it is

**Files:** [memory_module/chunking/document_chunker.py](memory_module/chunking/document_chunker.py), [memory_module/chunking/base_chunker.py](memory_module/chunking/base_chunker.py).

**Reframe:** The README is explicit: *"The chunker owns chunk metadata and chunk IDs. That is intentional. Chunking policy should live with the chunker, not the pipeline."* So the previous suggestion (extract a `chunk_assembly` module) **contradicts a load-bearing decision**. Withdrawn.

The friction I observed is still real, but the fix is different. Today `DocumentChunker.chunk(parsed_document, metadata)` reads from **two schemas**:

- `parsed_document.file_metadata.document_id` / `document_title` — parser's schema
- `metadata.get("tags")` — caller's loosely-typed dict

The first is a typed contract; the second is `Dict[str, Any]` with one undocumented expected key (the README even hedges: *"today the default chunker mainly expects `tags` from caller metadata"*). When a second chunker arrives, every chunker will independently re-discover what's allowed in that dict.

**Sketch (to grill):** introduce a typed `ChunkRequest` (or extend `DocumentParserResult` with caller-passed fields the pipeline merges in once):

```python
class ChunkRequest(BaseModel):
    parsed_document: DocumentParserResult
    tags: list[str] | None = None
    extra: dict[str, Any] = {}     # explicitly opaque escape hatch

class BaseChunker(ABC):
    def chunk(self, request: ChunkRequest) -> list[Chunk]: ...
```

Chunker still owns metadata composition (preserving the README's decision). What changes is that the chunker reads **one** typed object instead of one typed object + one dict-with-conventions. Tests build a `ChunkRequest`, not a `DocumentParserResult` plus folklore.

**Benefit:** Depth of the chunker interface goes up — every chunker faces the same input shape, the "tags" convention becomes explicit, and the caller metadata schema stops being a hidden contract leaking from `main.py` into every chunker.

---

### 5. Move format validation into each parser

**Files:** [memory_module/utils/file_operations.py](memory_module/utils/file_operations.py), [memory_module/parser/](memory_module/parser/).

**Problem:** `FileOps.validate(formats=[...])` knows MIME and probe logic for pdf/docx/csv/txt, but only `DocxParser.accepts()` calls it, only with `["docx"]`. The README says *"parsers own document understanding"* — validation is part of that. Centralizing it in a utility creates a third place a `PdfParser` author has to look.

**Sketch:** each parser's `accepts(file)` owns its extension + magic-byte + library-probe checks. `FileOps` shrinks to whatever is genuinely cross-parser (e.g. a hash helper) or disappears.

**Benefit:** Adding a parser stops requiring edits to `FileOps`. Validation lives next to the parser that knows the format. Aligns with the README's stated mindset.

---

## Summary of changes vs. the previous draft

| # | Previous suggestion | Revised suggestion | Why it changed |
|---|---|---|---|
| 1 | Delete `SimilarityRetrievalStrategy` shim | Keep the seam, **widen its interface** to carry `query_text` + richer signal | README names retrieval as a plugin point; the shim is the landing pad for the second adapter |
| 2 | Delete or wire `Storage` | Same — pick one | Unchanged; README is silent on this one |
| 3 | Consolidate 5 factories into one registry | Keep the layout; only remove the duplicate `_strategy`/`_backend` aliases in retrieval_factory | "One file per plugin point" mirrors the framework's documented layering |
| 4 | Extract a `chunk_assembly` module out of the chunker | Keep ownership in chunker; **normalize the chunker's input** to a single typed `ChunkRequest` | README says chunker ownership of metadata is intentional |
| 5 | Move validation into each parser | Same | README: *"parsers own document understanding"* |

---

## Open decisions before any of this lands

1. **Storage:** wire as the upload→parse seam, or delete? README is silent — needs an explicit call.
2. **Retrieval interface shape:** is `query_text` the right addition, or do you want strategies to also see candidate scores / embeddings (i.e. a two-phase `candidates → rerank` interface)?
3. **Chunker input:** extend `DocumentParserResult` vs. introduce `ChunkRequest`? The former keeps the parser as the single typed input; the latter keeps caller-passed fields out of the parser's contract.
4. **Aliases in `retrieval_factory`:** is `_strategy` or `_backend` the canonical term across the framework? (README uses *strategy* throughout — likely answer.)
