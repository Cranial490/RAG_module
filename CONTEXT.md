# RAG Module

A configurable Retrieval-Augmented Generation pipeline. Each pipeline stage (parser, chunker, embedder, vector DB, retrieval) is a swappable plugin point.

## Language

**Chunk**:
A piece of indexed content with its own metadata, embedding, and identity. Identity is independent of any query.
_Avoid_: Document, passage, fragment

**ScoredChunk**:
A **Chunk** paired with the score it earned against a specific query. Scores are only meaningful in the context of a query, so they live on this wrapper rather than on **Chunk** itself.
_Avoid_: RankedChunk, Result, Hit

**Parser Rejection**:
A parser's pre-flight refusal to handle a document, distinct from a parser failure during conversion. Surfaced by `parser.accepts(document)` returning False before any parsing work begins. A parser rejects a document type it does not support; a parser fails when it accepted a document and then could not parse it.
_Avoid_: Parser error, unsupported document

## Relationships

- A **ScoredChunk** wraps exactly one **Chunk**
- The vector DB and retrieval strategies traffic in **ScoredChunk**; everything else (chunker, embedder, indexing path) traffics in **Chunk**

## Example dialogue

> **Dev:** "When the re-ranker fuses dense and cross-encoder scores, does it return new **Chunks**?"
> **Domain expert:** "No — it returns the same **Chunks**, but as new **ScoredChunks** with a fused score. The chunk identity doesn't change; only its rank against the query does."
