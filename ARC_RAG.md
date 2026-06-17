# RAG Architecture - Knowledge, Indexing, and Hybrid Retrieval

This document explains the Retrieval-Augmented Generation branch of the app.

## Purpose

RAG gives the RCA and chat layers maintenance knowledge instead of making them
answer from model scores alone.

```text
SOPs / manuals / work orders / images
  -> parse and chunk
  -> text vectors + visual vectors + metadata
  -> hybrid retrieval
  -> LLM/fallback RCA and chat answers
```

## Main Files

```text
app/backend/rag_runtime.py
app/backend/llm_rca_runtime.py
app/backend/server.py
app/frontend/rag.html
app/frontend/index.html
```

RAG storage folder:

```text
app/backend/knowledge_rag/
```

## Knowledge Types Present

Base knowledge:

```text
app/backend/knowledge_rag/maintenance_knowledge.jsonl
```

Uploaded knowledge:

```text
app/backend/knowledge_rag/uploaded_documents.jsonl
app/backend/knowledge_rag/uploads/
```

Indexes:

```text
text_vector_index.jsonl
visual_index.jsonl
work_order_index.jsonl
```

Supported document types:

- text files: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.log`
- PDFs: `.pdf`
- spreadsheets/work orders: `.xlsx`, `.xls`
- HTML pages: `.html`, `.htm`
- CAD/drawing text: `.svg`, `.dxf`
- images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`

Base knowledge examples:

- safety SOPs
- gearbox inspection guidance
- crack inspection guide
- oil leak/lubrication guide
- corrosion assessment guide

Uploaded knowledge examples:

- C-MAPSS readme
- damage propagation PDF
- corrosion SOPs
- image SOPs with OCR/layout metadata
- work-order/spare-part Excel sheets

## Upload Endpoint

Documents are uploaded from the RAG page:

```text
POST /api/rag/documents
GET  /api/rag/documents
```

The upload endpoint stores the file and creates searchable chunks.

## Parsing Layer

The lightweight parser does this:

```text
file upload
  -> detect file extension
  -> extract text / metadata
  -> chunk text
  -> write JSONL records
  -> update indexes
```

Parsing behavior:

- PDFs use PyMuPDF when installed.
- Spreadsheets use pandas/openpyxl and also create work-order records.
- HTML uses BeautifulSoup when available.
- Images use optional OCR, layout analysis, and DINOv2 visual summary.
- CAD text support extracts searchable text/metadata, not full geometric
  reasoning.

## Image SOP Handling

For uploaded image documents:

```text
image SOP
  -> optional OCR
  -> OpenCV layout analysis
  -> DINOv2 image summary
  -> DINOv2 global visual embedding
  -> visual_index.jsonl
```

This lets chat/retrieval match visually similar documents, not only text.

## Text Embeddings

The current local implementation uses lightweight hashing vectors:

```text
local_hashing_384
```

Stored in:

```text
text_vector_index.jsonl
```

This keeps the app self-contained. External vector databases can be added later,
but the current app does not require Pinecone/ChromaDB to run.

## Visual Embeddings

DINOv2 global image embeddings are stored in:

```text
visual_index.jsonl
```

At query time, if the current scenario has an image path, the backend computes a
DINOv2 embedding for that image and compares it with stored image-document
embeddings.

## Hybrid Retrieval

Main function:

```text
rag_runtime.retrieve(payload, telemetry, vision)
```

The retrieval query is built from:

- scenario description
- asset type
- location
- operating state
- telemetry severity
- predicted RUL
- failure risk
- vision anomaly state
- predicted defect type

Scoring combines:

```text
TF-IDF semantic score
+ local vector score
+ keyword overlap score
+ code/id boost
+ defect tag boost
+ DINOv2 visual similarity score
```

Output:

```json
{
  "mode": "local_vector_tfidf_keyword_visual_hybrid",
  "results": [
    {
      "document_id": "GUIDE-LUB-005",
      "title": "Oil Leak and Lubrication Loss",
      "text": "Dark wet residue below a seal...",
      "score": 0.31,
      "visual_score": 0.0,
      "keyword_score": 0.2
    }
  ]
}
```

## Work Order Search

Work-order/spare-part data is indexed separately:

```text
work_order_index.jsonl
```

Endpoint:

```text
GET  /api/work-orders/search
POST /api/work-orders/search
```

It supports questions like:

```text
which part was damaged?
what spare part is needed for crack damage?
search work orders for bearing wear
```

## RCA Use

During `/api/infer`:

```text
telemetry output
+ vision output
+ RAG retrieved passages
-> llm_rca_runtime.generate_llm_rca
```

The RCA layer uses retrieved documents to format:

- root cause
- RCA status
- confidence
- next step
- citation IDs

The frontend no longer dumps raw retrieved passages. It uses chat and clean RCA
cards instead.

## Current Status

```text
base JSONL knowledge: implemented
document upload: implemented
PDF parsing: implemented when PyMuPDF is installed
spreadsheet/work-order parsing: implemented
image OCR: optional, depends on pytesseract
image layout analysis: implemented
DINOv2 visual indexing: implemented
local text vector index: implemented
hybrid retrieval: implemented
external Pinecone/ChromaDB runtime: optional/not required
separate reranker model: not implemented
```
