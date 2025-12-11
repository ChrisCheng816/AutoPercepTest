from pathlib import Path

KITI_DIR = Path("labels_kiti")
MAPPING = {
    "Car": 2,
    "Pedestrian": 0,
    "Truck": 2,
    "Van": 2,
    "Person_sitting": 0,
}


def get_kiti_labels():
    label_files = sorted(KITI_DIR.glob("*.txt"))
    all_labels = []
    for index, label_file in enumerate(label_files):
        labels = load_kiti_bboxes(label_file, index)
        all_labels.append((labels,index))
    return all_labels

def load_kiti_bboxes(label_path, index):
    results = []    
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            obj_type = parts[0].strip()
            if obj_type not in MAPPING:
                continue
            x_min = float(parts[4])
            y_min = float(parts[5])
            x_max = float(parts[6])
            y_max = float(parts[7])

            results.append({
                "type": MAPPING[obj_type],
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max
            })
    return results