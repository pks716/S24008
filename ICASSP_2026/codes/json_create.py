"""
Generate dataset JSON files for SynthRAD2025 Task1 anatomies.
Matches the format of synthrad23_task1_brain.json exactly.

Usage:
    python make_synthrad25_jsons.py

Edit DATA_ROOT and OUT_DIR to match your paths before running.
"""

import json
import os

# ── EDIT THESE ────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/pks/Desktop/icassp_code/data/SynthRad2025/Task2"
OUT_DIR   = "/home/pks/Desktop/icassp_code/data/meta/"

# Relative prefix used inside the JSON (matches how brain JSON uses "synthrad23/...")
# Set to empty string "" if you want absolute paths instead
JSON_PREFIX = "synthrad25/train/Task2"
# ─────────────────────────────────────────────────────────────────────────────

ANATOMY_NAMES = {
    "AB": "ab",   # Abdomen
    "HN": "hn",   # Head and Neck
    "TH": "th",   # Thorax
}

def make_json_for_anatomy(anatomy_code: str, anatomy_tag: str):
    """
    Scan DATA_ROOT/anatomy_code for case folders and build JSON.
    anatomy_code : folder name, e.g. "AB"
    anatomy_tag  : short name for source field, e.g. "ab"
    """
    anatomy_dir = os.path.join(DATA_ROOT, anatomy_code)

    if not os.path.isdir(anatomy_dir):
        print(f"  WARNING: {anatomy_dir} not found — skipping")
        return None

    # Collect all case folders, sorted for reproducibility
    case_ids = sorted([
        d for d in os.listdir(anatomy_dir)
        if os.path.isdir(os.path.join(anatomy_dir, d))
    ])

    if not case_ids:
        print(f"  WARNING: no case folders found in {anatomy_dir}")
        return None

    files = []
    missing = []
    for case_id in case_ids:
        case_dir = os.path.join(anatomy_dir, case_id)

        # Build relative paths matching JSON_PREFIX convention
        rel_base = f"{JSON_PREFIX}/{anatomy_code}/{case_id}"

        entry = {
            "mr":   f"{rel_base}/mr.mha",
            "ct":   f"{rel_base}/ct.mha",
            "mask": f"{rel_base}/mask.mha",
        }

        # Check which files actually exist (handle .mha vs .nii.gz)
        for key in ["mr", "ct", "mask"]:
            abs_mha  = os.path.join(case_dir, f"{key}.mha")
            abs_niigz = os.path.join(case_dir, f"{key}.nii.gz")
            if os.path.exists(abs_mha):
                entry[key] = f"{rel_base}/{key}.mha"
            elif os.path.exists(abs_niigz):
                entry[key] = f"{rel_base}/{key}.nii.gz"
            else:
                missing.append(f"{case_id}/{key}")

        files.append(entry)

    if missing:
        print(f"  NOTE: missing files for {anatomy_code}: {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}")

    dataset = {
        "source":   f"synthrad25_task2_{anatomy_tag}",
        "files":    files,
        "case_ids": case_ids,
    }

    return dataset


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for anatomy_code, anatomy_tag in ANATOMY_NAMES.items():
        print(f"\nProcessing {anatomy_code} ({anatomy_tag})...")

        dataset = make_json_for_anatomy(anatomy_code, anatomy_tag)
        if dataset is None:
            continue

        n = len(dataset["case_ids"])
        print(f"  Found {n} cases: {dataset['case_ids'][:3]}{'...' if n > 3 else ''}")

        out_path = os.path.join(OUT_DIR, f"synthrad25_task2_{anatomy_tag}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"  Saved: {out_path}")

        # Quick sanity check — print first entry
        first = dataset["files"][0]
        print(f"  First entry:")
        for k, v in first.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()