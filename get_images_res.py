from ultralytics import YOLO
from pathlib import Path

KITI_DIR = Path("image_kiti")

MAPPING = {
    "2": 2,
    "3": 2,
    "5": 2,
    "6": 2,
    "7": 2,
    "0": 0
}

model = YOLO('yolov8n.pt')

png_files = sorted(KITI_DIR.glob("*.png"))

def kiti_images_res(MIN_COUNT, MAX_COUNT):
    kiti_dict_overall=[]
    for img_path in png_files[MIN_COUNT:MAX_COUNT]:
        results = model(str(img_path))
        boxes_list = results[0].boxes.xyxy.tolist()
        shape_list = results[0].boxes.cls.tolist()
        kiti_dict = []
        for i in range(len(boxes_list)):
            if str(int(shape_list[i])) not in MAPPING:
                print("Unknown class:", shape_list[i])
                continue
            kiti_dict.append({"type": MAPPING[str(int(shape_list[i]))], "xmin": boxes_list[i][0], "ymin": boxes_list[i][1],"xmax": boxes_list[i][2], "ymax": boxes_list[i][3]})
        kiti_dict_overall.append(kiti_dict)

    return kiti_dict_overall