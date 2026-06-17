from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from model_config import config_value, resolve_app_path

DEFAULT_VISION_MODEL_DIR = Path(__file__).resolve().parent / "vision_dinov2"
VISION_MODEL_DIR = resolve_app_path(config_value("vision", "model_dir"), DEFAULT_VISION_MODEL_DIR)
DINO_LOCAL_DIR = resolve_app_path(
    config_value("vision", "dinov2_local_dir"),
    VISION_MODEL_DIR / "facebook_dinov2_base",
)
SIMILARITY_CHUNK_SIZE = 5000
TOP_PATCH_FRACTION = 0.05
_DINO_CACHE: dict[str, Any] = {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def default_demo_package(repo_root: Path) -> Path:
    return (repo_root / "app").resolve()


def vision_metadata() -> dict[str, Any]:
    return load_json(VISION_MODEL_DIR / str(config_value("vision", "metadata_file", "metadata.json")))


def load_dinov2_runtime() -> dict[str, Any]:
    if _DINO_CACHE:
        return _DINO_CACHE

    memory_path = VISION_MODEL_DIR / str(config_value("vision", "normal_patch_memory_file", "normal_patch_memory.pt"))
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)

    state = torch.load(memory_path, map_location="cpu")
    model_name = state.get("model_name", "facebook/dinov2-base")
    model_source = DINO_LOCAL_DIR if DINO_LOCAL_DIR.is_dir() else model_name
    image_size = int(state.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(
        model_source,
        size={"height": image_size, "width": image_size},
        crop_size={"height": image_size, "width": image_size},
        use_fast=False,
        local_files_only=DINO_LOCAL_DIR.is_dir(),
    )
    model = AutoModel.from_pretrained(
        model_source,
        local_files_only=DINO_LOCAL_DIR.is_dir(),
    ).to(device)
    model.eval()

    _DINO_CACHE.update(
        {
            "memory_bank": state["memory_bank"].float().contiguous(),
            "model_name": model_name,
            "model_source": str(model_source),
            "image_size": image_size,
            "grid_height": int(state.get("grid_height", image_size // int(state.get("patch_size", 14)))),
            "grid_width": int(state.get("grid_width", image_size // int(state.get("patch_size", 14)))),
            "normal_memory_sources": state.get("normal_memory_sources", []),
            "evaluation_mode": state.get("evaluation_mode", "functional_demo_no_independent_normal_test"),
            "processor": processor,
            "model": model,
            "device": device,
        }
    )
    return _DINO_CACHE


def infer_fault_from_path(image_path: str | None, fallback: str = "visual_anomaly") -> str:
    if not image_path:
        return fallback

    lowered = image_path.lower()
    if any(token in lowered for token in ("crack", "fracture", "split")):
        return "crack"
    if any(token in lowered for token in ("corrosion", "corossion", "corosion", "rust", "oxidation")):
        return "corrosion"
    if any(token in lowered for token in ("oil", "leak", "lubricant", "grease")):
        return "oil_leak"
    if any(token in lowered for token in ("wear", "abrasion", "scratch", "scoring")):
        return "wear"
    if any(token in lowered for token in ("heat", "overheat", "overheating", "burn", "thermal")):
        return "overheating"
    return fallback


def heatmap_path_for(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "vision_dinov2_result.png"


def extract_patch_embeddings(image: Image.Image, runtime: dict[str, Any]) -> torch.Tensor:
    inputs = runtime["processor"](images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(runtime["device"])
    with torch.inference_mode():
        outputs = runtime["model"](pixel_values=pixel_values)
        patches = F.normalize(outputs.last_hidden_state[:, 1:, :].float(), dim=-1)
    return patches[0].cpu()


def nearest_memory_distance(query_patches: torch.Tensor, runtime: dict[str, Any]) -> torch.Tensor:
    device = runtime["device"]
    memory_bank = runtime["memory_bank"]
    query_patches = query_patches.to(device)
    best_similarity = torch.full((len(query_patches),), -1.0, device=device)
    for start in range(0, len(memory_bank), SIMILARITY_CHUNK_SIZE):
        memory_chunk = memory_bank[start:start + SIMILARITY_CHUNK_SIZE].to(device)
        similarity = query_patches @ memory_chunk.T
        best_similarity = torch.maximum(best_similarity, similarity.max(dim=1).values)
    return (1.0 - best_similarity).clamp(min=0).cpu()


def score_from_patches(patch_scores: torch.Tensor) -> float:
    top_count = max(1, int(math.ceil(len(patch_scores) * TOP_PATCH_FRACTION)))
    return float(torch.topk(patch_scores, top_count).values.mean())


def dinov2_global_embedding(image_path: Path) -> list[float]:
    runtime = load_dinov2_runtime()
    image = Image.open(image_path).convert("RGB")
    patches = extract_patch_embeddings(image, runtime)
    embedding = F.normalize(patches.mean(dim=0), dim=0)
    return embedding.numpy().astype(float).tolist()


def upsample_heatmap(patch_map: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(patch_map)[None, None].float()
    return F.interpolate(
        tensor,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


def normalized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    high = float(np.percentile(heatmap, 99))
    low = float(np.min(heatmap))
    denom = max(high - low, 1e-9)
    return np.clip((heatmap - low) / denom, 0.0, 1.0)


def titled_panel(image: Image.Image, title: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height + 28), "white")
    panel.paste(image.resize((width, height), Image.Resampling.LANCZOS), (0, 28))
    cv_panel = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    cv2.putText(cv_panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(cv_panel, cv2.COLOR_BGR2RGB))


def save_dinov2_result_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    predicted_mask: np.ndarray,
    score: float,
    predicted_fault: str,
    output_path: Path,
) -> None:
    width, height = image.size
    heat_norm = normalized_heatmap(heatmap)
    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)
    colored_rgb = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    mask_rgb = Image.fromarray((predicted_mask.astype(np.uint8) * 255)).convert("RGB")

    rgb = np.asarray(image.convert("RGB"))
    jet = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.62, jet, 0.38, 0)
    overlay_rgb = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    panel_w = 320
    panel_h = int(panel_w * height / max(width, 1))
    panels = [
        titled_panel(image.convert("RGB"), predicted_fault, panel_w, panel_h),
        titled_panel(mask_rgb, "Predicted Mask", panel_w, panel_h),
        titled_panel(colored_rgb, "DINOv2 Heatmap", panel_w, panel_h),
        titled_panel(overlay_rgb, f"score={score:.4f}", panel_w, panel_h),
    ]
    gap = 20
    canvas = Image.new("RGB", (panel_w * 4 + gap * 3, panel_h + 28), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * (panel_w + gap), 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def score_image(image_path: Path, output_path: Path | None, predicted_fault: str) -> dict[str, Any]:
    metadata = vision_metadata()
    runtime = load_dinov2_runtime()
    image = Image.open(image_path).convert("RGB")
    patches = extract_patch_embeddings(image, runtime)
    patch_scores = nearest_memory_distance(patches, runtime)
    patch_map = patch_scores.reshape(runtime["grid_height"], runtime["grid_width"]).numpy()
    heatmap = upsample_heatmap(patch_map, image.size)
    score = score_from_patches(patch_scores)
    threshold = float(metadata.get("image_threshold", 0.008287952281534672))
    pixel_threshold = float(metadata.get("pixel_threshold", 0.057987332344055176))
    predicted_mask = (heatmap >= pixel_threshold).astype(np.uint8)

    if output_path is not None:
        save_dinov2_result_figure(
            image,
            heatmap,
            predicted_mask,
            score,
            predicted_fault,
            output_path,
        )

    return {
        "anomaly_score": score,
        "is_anomaly": bool(score >= threshold),
        "threshold": threshold,
        "pixel_threshold": pixel_threshold,
        "image_size": [int(image.width), int(image.height)],
        "model": runtime["model_name"],
        "model_source": runtime["model_source"],
        "normal_memory_sources": runtime["normal_memory_sources"],
        "evaluation_mode": runtime["evaluation_mode"],
        "patch_grid": [runtime["grid_height"], runtime["grid_width"]],
    }


def build_vision_result(
    payload: dict[str, Any],
    demo_package: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    image_path = payload.get("image_path") or payload.get("vision", {}).get("image_path")
    image_file = Path(image_path).expanduser().resolve() if isinstance(image_path, str) else None
    predicted_fault = infer_fault_from_path(image_path, "visual_anomaly")

    model_mode = "dinov2_patch_memory"
    model_error = ""
    heatmap = heatmap_path_for(run_dir)
    metrics = {
        "anomaly_score": 0.0,
        "is_anomaly": False,
        "threshold": 0.35,
        "image_size": [],
    }
    if image_file is None or not image_file.is_file():
        model_mode = "missing_image"
        model_error = "No readable image file was provided."
    else:
        try:
            metrics = score_image(image_file, heatmap, predicted_fault)
        except Exception as exc:
            model_mode = "image_processing_failed"
            model_error = f"{type(exc).__name__}: {exc}"

    return {
        "scenario_id": payload.get("scenario_id", "SCENARIO-001"),
        "asset_id": payload.get("asset_id", "ASSET-001"),
        "anomaly_score": metrics["anomaly_score"],
        "is_anomaly": metrics["is_anomaly"],
        "predicted_fault": predicted_fault,
        "severity": "warning" if metrics["is_anomaly"] else "normal",
        "location": "generated_image_heatmap",
        "image_path": str(image_file) if image_file else image_path,
        "heatmap_path": str(heatmap) if heatmap else None,
        "evidence": {
            "model_mode": model_mode,
            "model_error": model_error,
            "model": metrics.get("model"),
            "model_source": metrics.get("model_source"),
            "image_threshold": metrics["threshold"],
            "pixel_threshold": metrics.get("pixel_threshold"),
            "threshold": metrics["threshold"],
            "image_size": metrics["image_size"],
            "normal_memory_sources": metrics.get("normal_memory_sources", []),
            "evaluation_mode": metrics.get("evaluation_mode"),
            "patch_grid": metrics.get("patch_grid", []),
        },
        "limitations": [
            "DINOv2 detects and localizes anomalies but does not classify defect type.",
            "Fault labels are inferred from supplied file/context names unless a classifier is added.",
            "The normal memory bank was built from the synthetic mechanical demo data.",
        ],
    }


def build_rca_result(payload: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
    rag = payload.get("rag") if isinstance(payload.get("rag"), dict) else {}
    rag_results = rag.get("results") if isinstance(rag.get("results"), list) else []
    citation_ids = [
        str(result.get("document_id"))
        for result in rag_results
        if result.get("document_id")
    ]
    telemetry_risk = float(
        telemetry.get("failure_risk", payload.get("telemetry_risk", 0.0))
    )
    telemetry_rul = float(telemetry.get("predicted_rul", payload.get("predicted_rul", 125.0)))
    vision_anomaly = bool(vision.get("is_anomaly"))
    vision_score = float(vision.get("anomaly_score", 0.0))

    if telemetry_risk >= 0.75 and vision_anomaly:
        status = "critical"
        root_cause = "Telemetry degradation aligns with a strong visual anomaly."
        confidence = min(0.95, 0.7 * telemetry_risk + 0.3 * vision_score)
    elif vision_anomaly:
        status = "warning"
        root_cause = "A visual surface anomaly is present without corroborating critical telemetry."
        confidence = min(0.85, 0.5 * telemetry_risk + 0.5 * vision_score + 0.1)
    elif telemetry_risk >= 0.5:
        status = "warning"
        root_cause = "Telemetry suggests elevated risk before a visual defect is confirmed."
        confidence = min(0.8, telemetry_risk + 0.1)
    else:
        status = "normal"
        root_cause = "No strong telemetry or vision anomaly is present."
        confidence = max(0.2, 1.0 - telemetry_risk)

    if status == "critical":
        action_text = "Stop the asset and inspect the localized visual defect region."
    elif status == "warning":
        action_text = "Inspect the anomaly region and compare it with a clean reference before assigning a fault class."
    else:
        action_text = "Continue monitoring and re-evaluate on the next cycle or image capture."

    if rag_results:
        top_doc = rag_results[0]
        action_text = f"{action_text} Consult {top_doc.get('document_id')} ({top_doc.get('title')})."

    actions = [{
        "priority": 1,
        "action": action_text,
        "citations": citation_ids[:3],
    }]

    return {
        "scenario_id": payload.get("scenario_id", vision.get("scenario_id", "SCENARIO-001")),
        "asset_id": payload.get("asset_id", vision.get("asset_id", "ASSET-001")),
        "status": status,
        "root_cause": root_cause,
        "fault_location": vision.get("location"),
        "confidence": confidence,
        "evidence": [
            {
                "source": "telemetry",
                "observation": f"Predicted RUL is {telemetry_rul:.1f} cycles with failure risk {telemetry_risk:.3f}.",
            },
            {
                "source": "vision",
                "observation": f"Vision branch indicates {vision.get('predicted_fault')} with score {vision_score:.3f}.",
            },
            {
                "source": "rag",
                "observation": (
                    f"Retrieved {len(rag_results)} maintenance passages: "
                    f"{', '.join(citation_ids[:3]) or 'none'}."
                ),
            },
        ],
        "recommended_actions": actions,
        "citations": citation_ids,
        "alternative_hypotheses": [
            "Benign surface variation",
            "Localized wear or damage",
            "Vision-domain false positive",
        ],
        "limitations": vision.get("limitations", []) + [
            "RAG passages are advisory maintenance references, not autonomous instructions.",
        ],
        "safety": {
            "advisory_only": True,
            "autonomous_control": False,
            "qualified_person_required": True,
        },
    }


def predict_vision(
    payload: dict[str, Any],
    demo_package: Path,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    return build_vision_result(payload, demo_package, run_dir)


def fuse_results(telemetry: dict[str, Any], vision: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return build_rca_result({"telemetry": telemetry, **payload}, vision)
