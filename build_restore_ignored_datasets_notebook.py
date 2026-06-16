import json
from pathlib import Path


OUTPUT = Path("restore_ignored_datasets.ipynb")


def lines(text):
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    md(
        """
# Restore Ignored Datasets

This notebook recreates the ignored dataset folders in the same locations used
by the project:

```text
data/VisA/
data/CMAPSSData/
data/_cmapss_extract/
```

Run this notebook after cloning the repository on the Jupyter server. It
downloads public datasets, extracts them safely, normalizes the folder layout,
and validates all required files.
"""
    ),
    md("## 1. Paths and Configuration"),
    code(
        """
from pathlib import Path
import shutil
import tarfile
import zipfile
import urllib.request
import json

DATA_ROOT = Path("/workspace/notebooks/data")
CMAPSS_DIR = DATA_ROOT / "CMAPSSData"
CMAPSS_EXTRACT_DIR = DATA_ROOT / "_cmapss_extract"
VISA_DIR = DATA_ROOT / "VisA"

CMAPSS_URL = (
    "https://data.nasa.gov/api/views/ff5v-kuh6/files/"
    "7fca1b45-2e1e-4f72-9b5a-7b4f4f6d7b77?"
    "download=true&filename=CMAPSSData.zip"
)
CMAPSS_ZIP = DATA_ROOT / "cmapss_outer.zip"

VISA_URL = "https://amazon-visual-anomaly.s3.amazonaws.com/VisA_20220922.tar"
VISA_TAR = DATA_ROOT / "VisA_20220922.tar"

DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("Data root:", DATA_ROOT)
print("C-MAPSS target:", CMAPSS_DIR)
print("C-MAPSS extract folder:", CMAPSS_EXTRACT_DIR)
print("VisA target:", VISA_DIR)
"""
    ),
    md("## 2. Download Helper"),
    code(
        """
def download_if_missing(url, destination):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination}")
        return destination

    print(f"Downloading:\\n  {url}\\n  -> {destination}")
    urllib.request.urlretrieve(url, destination)
    print(
        f"Downloaded {destination.name}: "
        f"{destination.stat().st_size / (1024 * 1024):.2f} MB"
    )
    return destination
"""
    ),
    md("## 3. Restore C-MAPSS"),
    code(
        """
required_cmapss = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

download_if_missing(CMAPSS_URL, CMAPSS_ZIP)

CMAPSS_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(CMAPSS_ZIP, "r") as archive:
    archive.extractall(CMAPSS_EXTRACT_DIR)

candidate_files = {
    path.name: path
    for path in CMAPSS_EXTRACT_DIR.rglob("*")
    if path.is_file()
}

missing = [
    name for name in required_cmapss
    if name not in candidate_files
]

if missing:
    raise FileNotFoundError(
        "C-MAPSS extraction did not produce required files: "
        + ", ".join(missing)
    )

CMAPSS_DIR.mkdir(parents=True, exist_ok=True)
for name in required_cmapss:
    shutil.copy2(candidate_files[name], CMAPSS_DIR / name)

# Optional readme files if present.
for name in ["readme.txt", "README.md"]:
    if name in candidate_files:
        shutil.copy2(candidate_files[name], CMAPSS_DIR / name)

print("Restored C-MAPSS files:")
for name in required_cmapss:
    path = CMAPSS_DIR / name
    print(f"  {name}: {path.stat().st_size / 1024:.1f} KB")
"""
    ),
    md("## 4. Restore VisA"),
    code(
        """
download_if_missing(VISA_URL, VISA_TAR)

if not VISA_DIR.exists():
    print("Extracting VisA. This can take a few minutes.")
    with tarfile.open(VISA_TAR, "r") as archive:
        archive.extractall(DATA_ROOT, filter="data")
else:
    print("VisA folder already exists:", VISA_DIR)

# Some archives may extract into a nested folder. Normalize if needed.
if not VISA_DIR.exists():
    candidates = [
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name.lower().startswith("visa")
    ]
    if candidates:
        candidates[0].rename(VISA_DIR)

if not VISA_DIR.exists():
    raise FileNotFoundError("VisA folder was not created.")

print("VisA restored:", VISA_DIR)
"""
    ),
    md("## 5. Validate C-MAPSS"),
    code(
        """
cmapss_validation = []

for name in required_cmapss:
    path = CMAPSS_DIR / name
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    line_count = 0
    if exists:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line_count = sum(1 for _ in handle)
    cmapss_validation.append({
        "file": name,
        "exists": exists,
        "size_kb": round(size / 1024, 1),
        "lines": line_count,
    })

import pandas as pd

cmapss_validation = pd.DataFrame(cmapss_validation)
display(cmapss_validation)

if not cmapss_validation["exists"].all():
    raise ValueError("C-MAPSS validation failed.")
if (cmapss_validation["lines"] <= 0).any():
    raise ValueError("One or more C-MAPSS files are empty.")

print("C-MAPSS validation passed.")
"""
    ),
    md("## 6. Validate VisA"),
    code(
        """
visa_classes = sorted(
    path.name for path in VISA_DIR.iterdir()
    if path.is_dir()
)

expected_classes = {
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum",
}

print("VisA classes:", visa_classes)

missing_classes = expected_classes - set(visa_classes)
if missing_classes:
    raise ValueError(f"Missing VisA classes: {sorted(missing_classes)}")

visa_rows = []
for class_name in visa_classes:
    class_dir = VISA_DIR / class_name
    normal_dir = class_dir / "Data" / "Images" / "Normal"
    anomaly_dir = class_dir / "Data" / "Images" / "Anomaly"
    mask_dir = class_dir / "Data" / "Masks" / "Anomaly"

    normal_count = len(list(normal_dir.glob("*"))) if normal_dir.exists() else 0
    anomaly_count = (
        len(list(anomaly_dir.glob("*"))) if anomaly_dir.exists() else 0
    )
    mask_count = len(list(mask_dir.glob("*"))) if mask_dir.exists() else 0

    visa_rows.append({
        "class_name": class_name,
        "normal_images": normal_count,
        "anomaly_images": anomaly_count,
        "masks": mask_count,
    })

visa_summary = pd.DataFrame(visa_rows)
display(visa_summary)

if (visa_summary["normal_images"] == 0).any():
    raise ValueError("A VisA class has no normal images.")
if (visa_summary["anomaly_images"] == 0).any():
    raise ValueError("A VisA class has no anomaly images.")
if (visa_summary["masks"] == 0).any():
    raise ValueError("A VisA class has no masks.")

print("VisA validation passed.")
"""
    ),
    md("## 7. Save Restore Report"),
    code(
        """
report = {
    "data_root": str(DATA_ROOT),
    "cmapss_dir": str(CMAPSS_DIR),
    "cmapss_extract_dir": str(CMAPSS_EXTRACT_DIR),
    "visa_dir": str(VISA_DIR),
    "cmapss_files": cmapss_validation.to_dict(orient="records"),
    "visa_summary": visa_summary.to_dict(orient="records"),
    "status": "passed",
}

report_path = DATA_ROOT / "dataset_restore_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("Restore report:", report_path)
print("Dataset restore complete.")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(cells)} cells.")
