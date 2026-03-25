# Qdrant Vector DB RAG Pipeline

This project is a configurable Retrieval-Augmented Generation pipeline built around plugin strategies.

The short version:

- you upload a document
- a parser turns it into structured content
- a chunker decides how to split that content
- an embedder turns chunks into vectors
- a vector DB stores them
- a retrieval strategy fetches the most relevant chunks for a query

The real value is not just that it works once. The value is that each stage is swappable.

If you are building internal search, document QA, semantic memory, or domain-specific RAG systems, this project gives you a clean way to change one stage without rewriting the rest of the pipeline.

## Why This Exists

Most RAG systems start as a script:

1. read file
2. split text
3. embed text
4. push to vector DB
5. query vector DB

That works until the first real requirement lands:

- "we need DOCX and PDF support"
- "we want section-aware chunking"
- "we want a different embedder in staging vs production"
- "we want to test a different retrieval strategy"
- "we may move away from Qdrant later"

At that point, a direct script becomes brittle.

This project solves that by treating each pipeline stage as a strategy:

- parser strategy
- chunking strategy
- embedder strategy
- vector DB strategy
- retrieval strategy

Those strategies are resolved from config through factories, then orchestrated by `RAGPipeline`.

That design gives developers three important benefits:

- isolation: each stage has a clear job
- replaceability: you can swap implementations without editing pipeline code
- testability: each strategy can be tested in isolation or combined through the pipeline

## Core Idea: Strategy Plugins

Each major subsystem is a plugin point.

### Parser Strategies

Parser strategies take an uploaded file and return a structured `DocumentParserResult`.

Current example:

- `docx`

Why it matters:

- different document formats need different parsing logic
- richer parsers can return more than raw text
- chunkers can then use that structure instead of guessing from plain text

### Chunking Strategies

Chunking strategies take the parser output and turn it into normalized chunks.

Current example:

- `document`

Why it matters:

- some use cases want token chunking over full text
- some want section-based chunking
- some want domain-aware chunking rules

The chunker owns chunk metadata and chunk IDs. That is intentional. Chunking policy should live with the chunker, not the pipeline.

### Embedder Strategies

Embedder strategies convert text into vectors.

Current example:

- `azure_openai`

Why it matters:

- embedding providers change
- environments differ
- models change

You want the pipeline to ask for "an embedder", not care which provider is behind it.

### Vector DB Strategies

Vector DB strategies are responsible for persistence and retrieval from the underlying vector store.

Current example:

- `qdrant`

Why it matters:

- storage concerns are separate from parsing/chunking
- the rest of the system should not know about Qdrant-specific client details

### Retrieval Strategies

Retrieval strategies sit on top of the vector DB and define how query-time retrieval works.

Current example:

- `similarity`

Why it matters:

- today you may want pure similarity search
- tomorrow you may want hybrid retrieval, reranking, or custom filtering logic

The pipeline should ask for "a retrieval strategy", not hardcode one retrieval behavior forever.

## Architecture Overview

The system is split into a few layers.

### 1. Factories

Factories map simple config keys to concrete classes.

Examples:

- `parser_key = "docx"`
- `chunker_key = "document"`
- `embedder_key = "azure_openai"`
- `vector_db_key = "qdrant"`
- `retrieval_key = "similarity"`

This is what lets you initialize the pipeline from JSON instead of wiring concrete classes by hand.

### 2. Models

The project uses typed models to make stage boundaries explicit.

Important parser models:

- `FileMetadata`
- `ParsedSection`
- `ParsedContent`
- `DocumentParserResult`

Important chunking models:

- `ChunkMetadata`
- `Chunk`

These models are the contracts between stages.

### 3. `RAGPipeline`

`RAGPipeline` is the orchestrator.

It is not supposed to parse, chunk, embed, or retrieve by itself.

Its job is:

- resolve configured strategies
- run them in order
- move data from one stage to the next

That separation is what keeps the system maintainable.

### 4. APIs

The FastAPI app exposes the pipeline through HTTP:

- `/strategies`
- `/strategies/{component}`
- `/index`
- `/retrieve`

This gives you both a machine-consumable config discovery surface and operational endpoints for indexing and retrieval.

## Current Data Flow

### Indexing

1. Client sends config + file to `/index`
2. `RAGPipeline` resolves the configured strategies
3. Parser validates and parses the file
4. Parser returns structured content plus file metadata
5. Chunker reads parser output and creates chunks
6. Embedder generates vectors for each chunk
7. Vector DB stores the chunks and embeddings

### Retrieval

1. Client sends config + query to `/retrieve`
2. `RAGPipeline` resolves embedder, vector DB, and retrieval strategy
3. Embedder generates a query vector
4. Retrieval strategy uses the vector DB to fetch matching chunks
5. API returns the retrieved chunks

## Structured Parser Output

One of the most important design decisions in this project is that parsers do not just return a flat string anymore.

They return structured content.

Current parser contract:

```python
class ParsedSection(BaseModel):
    title: str | None
    text: str
    level: int | None
    metadata: dict[str, Any] | None


class ParsedContent(BaseModel):
    mode: Literal["text", "sections"]
    text: str
    sections: list[ParsedSection]


class DocumentParserResult(BaseModel):
    content: ParsedContent
    file_metadata: FileMetadata
```

Why this matters:

- simple parsers can still provide full plain text
- richer parsers can expose sections
- chunkers can choose behavior based on parser intent

This is a better design than forcing every parser into raw text.

## Why Developers Should Care

If you are a developer evaluating whether this pattern is worth the abstraction, the pitch is simple:

### You can evolve the system without rewriting it

If you want a smarter parser, you add a parser.
If you want a different chunker, you add a chunker.
If you want a different embedder, you change config.

You are not trapped in one monolithic ingestion script.

### You can experiment safely

A good RAG system usually emerges through iteration:

- chunking experiments
- retrieval experiments
- embedding model experiments
- metadata filter experiments

This project is set up for those experiments.

### You can explain the system to a team

The strategy boundaries are clear enough that different engineers can own different parts:

- one person can own parser work
- one can own chunking
- one can own infra/vector DB work
- one can own API integration

That is much harder in a tightly-coupled script.

## How To Use The Existing Strategies

## Available strategies

You can discover registered strategies through the API:

```bash
curl http://localhost:8000/strategies
```

Example response:

```json
{
  "parsers": {
    "selection_mode": "single",
    "strategies": ["docx"]
  },
  "chunkers": {
    "selection_mode": "single",
    "strategies": ["document"]
  },
  "embedders": {
    "selection_mode": "single",
    "strategies": ["azure_openai"]
  },
  "retrievals": {
    "selection_mode": "single",
    "strategies": ["similarity"]
  },
  "vector_dbs": {
    "selection_mode": "single",
    "strategies": ["qdrant"]
  }
}
```

Only one strategy can be selected per subsystem at a time.

### Minimal indexing config

```json
{
  "parser_key": "docx",
  "chunker_key": "document",
  "embedder_key": "azure_openai",
  "vector_db_key": "qdrant"
}
```

### Minimal retrieval config

```json
{
  "embedder_key": "azure_openai",
  "vector_db_key": "qdrant",
  "retrieval_key": "similarity"
}
```

## Index API

`/index` expects `multipart/form-data`.

Example:

```bash
curl -X POST http://localhost:8000/index \
  -F 'config={"parser_key":"docx","chunker_key":"document","embedder_key":"azure_openai","vector_db_key":"qdrant"}' \
  -F 'metadata={"tags":["finance","policy"]}' \
  -F 'file=@/absolute/path/to/sample.docx'
```

Notes:

- `config` must be a JSON object string
- `metadata` is optional
- today the default chunker mainly expects `tags` from caller metadata

## Retrieve API

`/retrieve` expects:

- `query` as a query parameter
- optional `top_k` as a query parameter
- `config` and optional `filters` in the JSON body

Example:

```bash
curl -X POST "http://localhost:8000/retrieve?query=finance%20summary&top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "embedder_key": "azure_openai",
      "vector_db_key": "qdrant",
      "retrieval_key": "similarity"
    },
    "filters": {
      "metadata.tags": "finance"
    }
  }'
```

## Filtering

Current retrieval filtering is passed through to the vector DB layer.

Example tag filter:

```json
{
  "filters": {
    "metadata.tags": "finance"
  }
}
```

This is designed to be simple first. You can evolve it into richer metadata filtering later.

## Running The Project

### Start Qdrant

```bash
docker compose up -d
```

### Start the API

```bash
uvicorn main:app --reload
```

### Strategy discovery

```bash
curl http://localhost:8000/strategies
```

### Optional dashboard

The FastAPI app also serves a dashboard if the static files are present:

- `GET /dashboard`

## How To Add A New Strategy

This is the part developers usually care about most.

### Add a new parser

1. implement `DocumentParserBase`
2. return `DocumentParserResult`
3. register the parser in `parser_factory.py`

Example flow:

```python
class PdfParser(DocumentParserBase):
    def accepts(self, file_stream: UploadFile) -> bool:
        ...

    def convert(self, file_stream: UploadFile) -> DocumentParserResult:
        ...
```

Then register:

```python
PARSER_REGISTRY["pdf"] = PdfParser
```

### Add a new chunker

1. implement `BaseChunker`
2. consume `DocumentParserResult`
3. return normalized `Chunk` objects
4. register it in `chunking_factory.py`

### Add a new embedder

1. implement `BaseEmbedder`
2. make `embed(text)` return a vector or a one-item batch that the pipeline can flatten
3. register it in `embedder_factory.py`

### Add a new vector DB backend

1. implement `BaseVectorMemory`
2. handle `add_chunks`, `retrieve`, and `delete`
3. register it in `vector_db_factory.py`

### Add a new retrieval strategy

1. implement `BaseRetrievalStrategy`
2. use `self.vector_db`
3. register it in `retrieval_factory.py`

That is the developer experience this project is aiming for:

- small interface
- one registration step
- config-driven activation

## Recommended Development Mindset

If you are extending this project, follow these rules:

- parsers own document understanding
- chunkers own chunk formation and chunk metadata
- embedders own vector generation
- vector DBs own persistence
- retrieval strategies own query-time retrieval behavior
- `RAGPipeline` should only orchestrate

If `RAGPipeline` starts accumulating parsing or chunking logic, the abstraction is drifting.

## What Is Stable vs Experimental

Stable enough to build on:

- factory-based strategy resolution
- indexing flow
- retrieval flow
- Qdrant storage path
- DOCX parser as baseline example

Still evolving:

- richer parser structures beyond sections
- more sophisticated metadata filters
- additional parsers/chunkers/embedders/vector DBs
- dashboard UX

## Testing

The project includes a top-level `tests/` suite focused on the current architecture:

- parser tests
- chunking tests
- pipeline tests
- factory tests
- vector DB tests
- storage tests
- API tests
- CLI tests

Run when your environment has test dependencies installed:

```bash
pytest tests -q
```

## Final Pitch

If you just want one hardcoded script, this project is more structure than you need.

If you want a RAG foundation that can survive changing requirements, this design pays off quickly.

The plugin strategy model is the point:

- it keeps experiments cheap
- it keeps responsibilities clean
- it lets your system grow without turning into a rewrite

That is the developer pitch:

use this when you want a RAG system that is meant to evolve.
