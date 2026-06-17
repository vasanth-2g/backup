# Chat Architecture - Scenario Grounded Maintenance Assistant

This document explains the chat UI added to the inference page.

## Purpose

The chat UI lets the operator ask questions after a scenario has run.

It answers from:

- current telemetry output
- current DINOv2 vision output
- retrieved RAG knowledge
- DINOv2 visual retrieval when scenario image is available

It should not answer from outside knowledge. If the answer is not supported, it
returns:

```text
Out of knowledge.
```

## Frontend Location

Main page:

```text
app/frontend/index.html
```

UI section:

```text
Scenario Chat
```

The chat appears only after scenario inference returns a result. The page no
longer displays raw JSON or a raw `Retrieved Knowledge` block.

## Backend Endpoint

```text
POST /api/chat
```

Request shape:

```json
{
  "question": "what should I do next?",
  "context": {
    "telemetry": {},
    "vision": {},
    "fusion": {},
    "run": {}
  }
}
```

Response shape:

```json
{
  "answer": "Next step: inspect the seal, fittings, drain points, and lubricant level...",
  "confidence": 0.71,
  "citations": ["GUIDE-LUB-005"],
  "retrieval_score": 0.31,
  "mode": "deterministic_next_steps"
}
```

## Backend Flow

Main files:

```text
app/backend/server.py
app/backend/rag_runtime.py
app/backend/llm_rca_runtime.py
```

Flow:

```text
chat question
  -> /api/chat
  -> load saved scenario.json from run folder
  -> combine question + scenario + telemetry + vision
  -> run hybrid RAG retrieval
  -> call LLM/fallback chat answer
  -> return human-readable answer
```

## Scenario Context

The chat endpoint uses the same scenario run produced by `/api/infer`.

It includes:

- scenario description
- predicted remaining cycles
- failure risk
- telemetry severity
- predicted vision fault
- processed/cropped image path when available

If a processed image path exists, it is used for DINOv2 visual retrieval.

## Retrieval Grounding

Chat retrieval uses:

```text
question
+ scenario description
+ telemetry output
+ vision output
+ current scenario image embedding
```

This makes questions like these grounded:

```text
what should I do next?
why is priority warning?
what does 129.5 cycles mean?
if usage is 10 cycles per day, how many days?
what part should be inspected for oil leak?
```

## Answer Modes

The backend has deterministic fallback modes so chat still works without a local
LLM model loaded.

Modes:

```text
deterministic_next_steps
deterministic_cycle_math
deterministic_rul_explanation
deterministic_rag_chat
huggingface_rag_chat
out_of_knowledge
```

Examples:

### Next Step Question

Question:

```text
what to do next?
```

Answer intent:

```text
inspect affected area
clean/check seal or component
confirm defect growth
monitor telemetry
```

### Cycle Conversion Question

Question:

```text
129.5 cycles remaining and 10 cycles per day means what?
```

Answer:

```text
129.5 / 10 = about 13 days
```

The answer also reminds the user that inspection should happen earlier when
vision shows a defect.

### Unsupported Question

Question:

```text
what is the weather tomorrow?
```

Answer:

```text
Out of knowledge.
```

## LLM Use

`llm_rca_runtime.py` can call a Hugging Face local model when available.

If the model is missing or unavailable, the deterministic fallback still gives
grounded answers for common maintenance questions.

The LLM/fallback is instructed to use only:

- telemetry result
- vision result
- retrieved knowledge

It should not invent citations.

## UI Behavior

The frontend:

- stores the latest inference result in memory
- sends it with each chat question
- shows answer bubbles
- shows concise confidence/source metadata
- hides raw retrieved passages
- hides raw backend JSON

## Current Status

```text
chat UI on inference page: implemented
/api/chat endpoint: implemented
scenario-aware RAG retrieval: implemented
DINOv2 visual query path: implemented
out-of-knowledge behavior: implemented
cycle math handling: implemented
next-step handling: implemented
external live chat memory across sessions: not implemented
```
