# Recreate Current Server Notebook State

This file is the recovery checklist for the `notebooks_backup` repository. Use
it when you delete local/server generated folders and need to rebuild the same
working state from code and notebooks.

## Repository Purpose

This repository stores the Jupyter notebooks and small contracts needed for the
Industrial AI demo:

```text
telemetrics/   C-MAPSS extraction, RUL baseline, risk baseline
vision/        VisA baseline, synthetic mechanical images, DINOv2 vision
knowledge/     synthetic maintenance documents and RAG artifacts
data/          runtime datasets only, mostly ignored by Git
```

The active baseline does not fine-tune an LLM. The current flow is:

```text
XGBoost telemetry
  + DINOv2 visual anomaly localization
  + hybrid RAG
  -> evidence fusion
  -> deterministic RCA fallback
  -> optional pretrained reasoning model later
```

## What Git Should Keep

Keep notebooks and small contracts:

```text
telemetrics/*.ipynb
vision/*.ipynb
knowledge_rag.ipynb
knowledge/artifacts/rag/*.json
knowledge/artifacts/rag/*.jsonl
knowledge/artifacts/rag/*.csv
knowledge/artifacts/rag/maintenance_embeddings.npy
telemetrics/artifacts/cmapss/fd001/metadata.json
telemetrics/artifacts/cmapss/fd001/official_test_predictions.csv
telemetrics/artifacts/cmapss/fd001/risk/metadata.json
telemetrics/artifacts/cmapss/fd001/risk/official_test_risk_outputs.csv
telemetrics/artifacts/cmapss/fd001/risk/telemetry_contract_example.json
vision/artifacts/*/**/metadata.json
vision/artifacts/*/**/vision_contract_example.json
vision/artifacts/*/**/test_results.csv
vision/artifacts/*/**/official_test_results.csv
vision/artifacts/visa/visa_manifest.csv
vision/artifacts/visa/visa_validation_report.json
```

Do not rely on Git for full raw datasets or generated image folders.

## What Can Be Deleted and Regenerated

These folders/files are generated or downloaded. They can be deleted if disk is
needed, then recreated by running the notebooks below.

```text
data/CMAPSSData/
data/VisA/
data/_cmapss_extract/
data/cmapss_outer.zip
data/synthetic_mechanical/generated/anomaly/
data/synthetic_mechanical/generated/masks/
data/synthetic_mechanical/generated/normal/
data/synthetic_mechanical/generated/component_masks/
data/synthetic_mechanical/generated/dataset_preview.png
telemetrics/artifacts/
vision/artifacts/
knowledge/artifacts/
.ipynb_checkpoints/
```

Keep or re-provide these before regenerating synthetic mechanical data:

```text
data/synthetic_mechanical/source_normal/
```

That folder contains the clean normal mechanical source images. Without it, the
synthetic generator cannot create anomalies.

## Recommended `.gitignore`

Use this if you want Git to keep only code, notebooks, and small metadata:

```gitignore
data/VisA/
data/CMAPSSData/
data/_cmapss_extract/
data/*.zip

data/synthetic_mechanical/source_normal/
data/synthetic_mechanical/generated/anomaly/
data/synthetic_mechanical/generated/masks/
data/synthetic_mechanical/generated/normal/
data/synthetic_mechanical/generated/component_masks/
data/synthetic_mechanical/generated/dataset_preview.png

.ipynb_checkpoints/
**/.ipynb_checkpoints/
*.zip
*.pt
*.pth
```

## Server Folder Layout

Expected server root:

```text
/workspace/notebooks/
  data/
    CMAPSSData/
    VisA/
    synthetic_mechanical/
      source_normal/
      generated/
    knowledge/
  telemetrics/
  vision/
  knowledge/
  integration/
```

## Execution Order

Run notebooks in this order to recreate the current working state.

### 1. C-MAPSS Data Extraction

Notebook:

```text
/workspace/notebooks/telemetrics/Telemetries_data_extraction.ipynb
```

Run all cells.

Creates or validates:

```text
/workspace/notebooks/data/CMAPSSData/
```

Important sections:

```text
Load and Summarize All Subsets
FD001 Sanity Check
```

### 2. RUL Baseline

Notebook:

```text
/workspace/notebooks/telemetrics/rul_baseline.ipynb
```

Run through:

```text
13. Save Reproducible Artifacts
```

Creates:

```text
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/xgboost_fd001_rul.json
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/metadata.json
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/official_test_predictions.csv
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/figures/
```

Known successful baseline:

```text
Official test capped RMSE: about 16.05
```

### 3. Risk Baseline

Notebook:

```text
/workspace/notebooks/telemetrics/risk_baseline.ipynb
```

Run through:

```text
10. Save Artifacts
```

Creates:

```text
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/risk/xgboost_fd001_failure_classifier.json
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/risk/metadata.json
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/risk/official_test_risk_outputs.csv
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/risk/telemetry_contract_example.json
```

Known successful baseline:

```text
Test ROC AUC: about 0.9899
Test PR AUC: about 0.9679
```

### 4. VisA Data Preparation

Notebook:

```text
/workspace/notebooks/vision/btad_data_preparation.ipynb
```

Despite the filename, this notebook prepares VisA.

Run through:

```text
6. Save Manifest and Dataset Report
```

Creates:

```text
/workspace/notebooks/data/VisA/
/workspace/notebooks/vision/artifacts/visa/visa_manifest.csv
/workspace/notebooks/vision/artifacts/visa/visa_validation_report.json
/workspace/notebooks/vision/artifacts/visa/visa_dataset_summary.csv
```

Known validation:

```text
12 classes
9621 normal images
1200 anomaly images
1200 masks
0 missing files
```

### 5. VisA DINOv2 Baseline

Notebook:

```text
/workspace/notebooks/vision/patchD_load_ROCm.ipynb
```

Run through:

```text
12. Save Memory Bank, Results, and Vision Contract
```

Creates:

```text
/workspace/notebooks/vision/artifacts/visa/dinov2/pcb1/metadata.json
/workspace/notebooks/vision/artifacts/visa/dinov2/pcb1/official_test_results.csv
/workspace/notebooks/vision/artifacts/visa/dinov2/pcb1/vision_contract_example.json
/workspace/notebooks/vision/artifacts/visa/dinov2/pcb1/figures/
```

Optional large artifact:

```text
/workspace/notebooks/vision/artifacts/visa/dinov2/pcb1/normal_patch_memory.pt
```

This `.pt` file is large and can be regenerated. Do not push it unless you
intentionally want model memory banks in Git.

### 6. Synthetic Mechanical Defect Generation

Before running, place clean normal mechanical images here:

```text
/workspace/notebooks/data/synthetic_mechanical/source_normal/
```

Notebook:

```text
/workspace/notebooks/vision/synthetic_mechanical_defects.ipynb
```

Run through:

```text
4. Generate Images, Masks, and Manifest
5. Validate Generated Images and Masks
7. Save Dataset Report
```

Creates:

```text
/workspace/notebooks/data/synthetic_mechanical/generated/manifest.csv
/workspace/notebooks/data/synthetic_mechanical/generated/generation_report.json
/workspace/notebooks/data/synthetic_mechanical/generated/anomaly/
/workspace/notebooks/data/synthetic_mechanical/generated/masks/
/workspace/notebooks/data/synthetic_mechanical/generated/normal/
/workspace/notebooks/data/synthetic_mechanical/generated/component_masks/
```

Known successful generation:

```text
source_normal_images: 2
anomaly_samples: 200
invalid_samples: 0
```

### 7. Synthetic Mechanical DINOv2 Demo

Notebook:

```text
/workspace/notebooks/vision/synthetic_mechanical_dinov2.ipynb
```

Run through:

```text
11. Save Artifacts and Vision Contract
```

Creates:

```text
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/metadata.json
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/test_results.csv
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/vision_contract_example.json
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/figures/
```

Optional large artifact:

```text
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/normal_patch_memory.pt
```

Current contract should say:

```json
{
  "predicted_fault": "visual_anomaly",
  "severity": "unknown",
  "evaluation_mode": "functional_demo_no_independent_normal_test"
}
```

Do not describe this as a reliable crack/corrosion classifier.

### 8. Knowledge and RAG

Notebook:

```text
/workspace/notebooks/knowledge/knowledge_rag.ipynb
```

Run all cells, especially:

```text
2. Create the Versioned Synthetic Knowledge Pack
7. Validate Citations and Save the RAG Contract
```

Creates:

```text
/workspace/notebooks/data/knowledge/maintenance_knowledge.jsonl
/workspace/notebooks/knowledge/artifacts/rag/maintenance_embeddings.npy
/workspace/notebooks/knowledge/artifacts/rag/source_store.jsonl
/workspace/notebooks/knowledge/artifacts/rag/rag_contract_example.json
/workspace/notebooks/knowledge/artifacts/rag/metadata.json
/workspace/notebooks/knowledge/artifacts/rag/retrieval_evaluation.csv
```

Known successful retrieval:

```text
Recall@1: 1.0
Recall@3: 1.0
Recall@5: 1.0
MRR:      1.0
Mean latency: about 5.45 ms
```

### 9. Multimodal Fusion and Rule-Based RCA

Notebook:

```text
/workspace/notebooks/integration/note6_multimodal_fusion_rule_rca.ipynb
```

Run all cells, especially:

```text
8. Save the End-to-End Baseline Artifacts
```

Requires these existing contracts:

```text
/workspace/notebooks/telemetrics/artifacts/cmapss/fd001/risk/telemetry_contract_example.json
/workspace/notebooks/vision/artifacts/synthetic_mechanical/dinov2/vision_contract_example.json
/workspace/notebooks/knowledge/artifacts/rag/rag_contract_example.json
```

Creates:

```text
/workspace/notebooks/integration/artifacts/multimodal_rule_rca/normalized_evidence.json
/workspace/notebooks/integration/artifacts/multimodal_rule_rca/rule_rca_output.json
/workspace/notebooks/integration/artifacts/multimodal_rule_rca/reasoning_input.json
/workspace/notebooks/integration/artifacts/multimodal_rule_rca/execution_trace.json
```

## Minimum Files to Preserve Outside Git

If deleting generated folders, preserve at least:

```text
data/synthetic_mechanical/source_normal/
```

Everything else can be regenerated from the notebooks, although VisA and
sentence-transformer downloads require network access.

## Minimum Files to Push to Rebuild From Scratch

Push these:

```text
.gitignore
RECREATE_CURRENT_STATE.md
telemetrics/*.ipynb
vision/*.ipynb
knowledge_rag.ipynb
```

If available in the repo, also push:

```text
integration/*.ipynb
```

Small optional data/contracts:

```text
data/knowledge/maintenance_knowledge.jsonl
data/synthetic_mechanical/generated/manifest.csv
data/synthetic_mechanical/generated/generation_report.json
knowledge/artifacts/rag/*.json
knowledge/artifacts/rag/*.jsonl
knowledge/artifacts/rag/*.csv
```

## Files Not Worth Pushing

Do not push:

```text
data/VisA/
data/CMAPSSData/
data/_cmapss_extract/
data/*.zip
data/synthetic_mechanical/generated/anomaly/
data/synthetic_mechanical/generated/masks/
data/synthetic_mechanical/generated/normal/
data/synthetic_mechanical/generated/component_masks/
vision/artifacts/**/normal_patch_memory.pt
.ipynb_checkpoints/
```

## Push Checklist

Before pushing:

```bash
git status --short --ignored
git ls-files | grep -E "\\.pt$|data/VisA|data/CMAPSSData|_cmapss_extract|\\.zip$"
```

The second command should return nothing unless you intentionally track those
large files.

Then:

```bash
git add RECREATE_CURRENT_STATE.md .gitignore
git add telemetrics/*.ipynb vision/*.ipynb knowledge_rag.ipynb
git commit -m "Add reproducible notebook recovery runbook"
git push origin main
```

