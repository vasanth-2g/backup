# Vision Architecture - ROI Crop, DINOv2 Memory, and Defect Overlay

This document explains the current vision branch used by the app.

## Purpose

Vision detects visual defects from the uploaded inspection image and returns:

- whether a visual anomaly exists
- predicted defect label
- anomaly score
- marked defect-region image
- ROI/crop metadata

Vision is one RCA evidence source. It is combined with telemetry and RAG.

```text
inspection image
  -> mechanical asset detection/crop
  -> DINOv2 patch embedding
  -> compare with normal memory
  -> anomaly score + region mask
  -> transparent defect overlay
  -> RCA
```

## Models Used

Main file:

```text
app/backend/vision_runtime.py
```

Model folders:

```text
app/backend/vision_dinov2/facebook_dinov2_base
app/backend/vision_dinov2/google_owlvit_base_patch32
app/backend/vision_dinov2/facebook_sam_vit_base
```

Roles:

- DINOv2: frozen visual feature extractor.
- OWL-ViT: open-vocabulary object detection for robot/mechanical asset ROI.
- SAM: segmentation mask for the detected asset.

The app does not train DINOv2 during runtime.

## Normal Reference Memory

Normal/reference memory file:

```text
app/backend/vision_dinov2/normal_patch_memory.pt
```

This stores DINOv2 patch embeddings from clean normal images.

Important:

```text
Vision Memory upload = normal images only
Scenario image upload = inspection/anomaly test image
```

Do not upload anomaly images into Vision Memory. If anomaly images are added to
normal memory, the anomaly can become treated as normal and detection becomes
weak.

Memory update endpoint:

```text
GET  /api/vision/memory
POST /api/vision/memory
```

The RAG page has the upload panel for clean normal/reference images.

## Runtime Flow

```text
POST /api/infer
  -> save uploaded image in app/runtime/runs/<timestamp>/inputs/
  -> detect robot/mechanical asset with OWL-ViT
  -> segment detected asset with SAM
  -> remove background / crop asset
  -> run DINOv2 on processed crop
  -> compare image patches with normal_patch_memory.pt
  -> create anomaly mask
  -> create transparent defect overlay
  -> return vision result to RCA
```

If OWL-ViT/SAM cannot run, the backend falls back to OpenCV foreground crop.

ROI modes:

```text
owlvit_sam_roi_crop
foreground_roi_crop
rejected_no_mechanical_asset
```

If no robot/mechanical asset is detected, the scenario is rejected before
telemetry/RAG/RCA continue.

```text
non-mechanical image
  -> HTTP 422
  -> error = image_rejected
```

## DINOv2 Anomaly Detection

DINOv2 converts the processed image into patch embeddings.

```text
new image patch embedding
  -> compare against normal memory patch embeddings
  -> high distance = unusual patch
```

The patch distances become:

- image-level anomaly score
- pixel/patch threshold
- anomaly region mask
- transparent overlay image

## Defect Labels

Current fault labels are inferred from image filename or scenario context, not
from a trained classifier.

Supported labels include:

```text
oil_leak
corrosion
crack
wear
overheating
visual_anomaly
```

Future improvement: add a trained fault classifier so labels do not depend on
filename/context keywords.

## Overlay Output

The frontend displays the marked defect image in **Vision Defect Location**.

The image uses transparent color marking. It avoids drawing raw scores or text
labels on top of the image.

Color intent:

```text
oil_leak    -> blue
crack       -> red
corrosion   -> orange
wear        -> purple
overheating -> orange/red
unknown     -> teal
```

## Vision Output

Example:

```json
{
  "anomaly_score": 0.72,
  "is_anomaly": true,
  "predicted_fault": "oil_leak",
  "severity": "warning",
  "processed_image_path": "app/runtime/runs/.../outputs/vision_robot_roi.png",
  "heatmap_url": "/api/runtime/runs/.../outputs/vision_heatmap.png",
  "evidence": {
    "model_mode": "dinov2_patch_memory",
    "roi_mode": "owlvit_sam_roi_crop",
    "roi_detector_label": "industrial robot"
  }
}
```

## RCA Role

Vision can raise RCA priority even when telemetry is normal.

Example:

```text
Telemetry:
  failure risk low
  remaining cycles high

Vision:
  oil leak detected
  localized defect region marked

RCA:
  warning status
  inspect seal/lubrication area
  keep telemetry under monitoring
```

## Files

```text
app/backend/vision_runtime.py
app/backend/server.py
app/frontend/index.html
app/frontend/rag.html
app/model_config.json
app/robot_roi_before_dinov2.ipynb
```

## Current Status

```text
DINOv2 frozen embedding: implemented
normal patch memory: implemented
Vision Memory endpoint: implemented
OWL-ViT detection: implemented
SAM segmentation: implemented
OpenCV fallback crop: implemented
transparent defect overlay: implemented
non-mechanical image rejection: implemented
trained fault classifier: not implemented
```
