import os
import csv
import shutil
from pathlib import Path

# === CONFIGURATION ===
DATA_PATH = "/projects/prjs1635/datasets/fsd50k/FSD50K.eval_audio"
OUTPUT_DIR = "/projects/prjs1635/datasets/fsd50k/FSD50K.selected_eval"
AUDIO_EXT = ".wav"

# Path to eval CSV
eval_csv_default = "/projects/prjs1635/datasets/fsd50k/FSD50K.ground_truth/eval.csv"
EVAL_CSV = os.environ.get("EVAL_CSV", eval_csv_default)

# Path to included labels CSV
INCLUDED_CSV = "datasets/fsd50k_vocab.csv"


def read_included_labels(included_csv_path):
    included_labels = set()
    with open(included_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # skip empty rows
            if row:
                included_labels.add(row[0].strip().lower())
    print(f"[DEBUG] Included labels loaded: {included_labels}")
    return included_labels

def read_eval_csv(eval_csv_path):
    id_to_labels = {}
    with open(eval_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                # skip malformed rows
                continue
            audio_id = row[0].strip()
            labels = set(label.strip().lower() for label in row[1].split(',') if label.strip())
            id_to_labels[audio_id] = labels
    print(f"[DEBUG] Total audio IDs loaded from eval.csv: {len(id_to_labels)}")
    return id_to_labels

def main():
    included_labels = read_included_labels(INCLUDED_CSV)
    print(f"Loaded {len(included_labels)} included labels.")

    id_to_labels = read_eval_csv(EVAL_CSV)
    print(f"Loaded {len(id_to_labels)} audio IDs from eval.csv.")

    # Find audio IDs with at least one included label
    selected_ids = []
    for audio_id, labels in id_to_labels.items():
        intersection = labels & included_labels
        if intersection:
            selected_ids.append(audio_id)
        
    print(f"[DEBUG] Found {len(selected_ids)} audio files matching included labels.")
    if selected_ids:
        print(f"[DEBUG] First 10 selected IDs: {selected_ids[:10]}")
        for sid in selected_ids[:10]:
            print(f"[DEBUG] {sid} labels: {id_to_labels[sid]}")
    else:
        print("[DEBUG] No audio IDs matched the included labels. Check label formats and contents.")

    # Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy files
    missing_files = []
    for audio_id in selected_ids:
        src_file = Path(DATA_PATH) / f"{audio_id}{AUDIO_EXT}"
        dst_file = Path(OUTPUT_DIR) / f"{audio_id}{AUDIO_EXT}"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
        else:
            missing_files.append(str(src_file))

    print(f"Copied {len(selected_ids) - len(missing_files)} files to {OUTPUT_DIR}.")
    if missing_files:
        print(f"WARNING: {len(missing_files)} files were missing and not copied.")
        with open("missing_files.log", "w") as logf:
            for f in missing_files:
                logf.write(f"{f}\n")
        print("See missing_files.log for details.")

if __name__ == "__main__":
    main() 