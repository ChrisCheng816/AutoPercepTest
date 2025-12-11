import random
import numpy as np
import cv2
from perturbation import apply_brightness, apply_contrast, apply_fog, apply_blur
import json
from getLabels import get_kiti_labels
from tools import *
from ultralytics import YOLO
import logging
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
MAX_STEPS= 500

MAPPING = {
    "2": 2,
    "3": 2,
    "5": 2,
    "6": 2,
    "7": 2,
    "0": 0
}


def apply_all(
    image,
    b_brightness,
    c_contrast,
    f_fog,
    b_blur,
    sigma_max=10.0,
    fog_max=0.9,
    factor_min=0.1,
    factor_max=10.0,
    alpha_min=0.1,
    alpha_max=10.0,
):
    img = cv2.imread(image)
    img = img.astype(np.uint8)
    img = apply_brightness(img, b_brightness, factor_min=factor_min, factor_max=factor_max)
    img = apply_contrast(img, c_contrast, alpha_min=alpha_min, alpha_max=alpha_max)
    img = apply_fog(img, f_fog, fog_max=fog_max)
    img = apply_blur(img, b_blur, sigma_max=sigma_max)
    return img

def eval_condition(image, label, car_zero, person_zero, car_mean, person_mean):
    model = YOLO('yolov8n.pt')
    results = model(image)
    boxes_list = results[0].boxes.xyxy.tolist()
    shape_list = results[0].boxes.cls.tolist()
    kiti_dict = []
    for i in range(len(boxes_list)):
        if str(int(shape_list[i])) not in MAPPING:
            continue
        kiti_dict.append({"type": MAPPING[str(int(shape_list[i]))], "xmin": boxes_list[i][0], "ymin": boxes_list[i][1],"xmax": boxes_list[i][2], "ymax": boxes_list[i][3]})
    car_image_iou, person_image_iou = load_compute([kiti_dict], [label], 0, 1)
    car_image_iou, car_zero_num = delete_zero(car_image_iou)
    person_image_iou, person_zero_num = delete_zero(person_image_iou)
    car_mean_after, people_mean_after = compute_mean(car_image_iou[0], person_image_iou[0])
    if (car_mean_after is not None and car_mean_after < 0.6) or (people_mean_after is not None and people_mean_after < 0.4):
        return True, car_mean_after, people_mean_after, car_zero_num, person_zero_num
    if (car_mean_after is None and car_mean is not None)  or (people_mean_after is None and person_mean is not None):
        return True, car_mean_after, people_mean_after, car_zero_num, person_zero_num
    return False, None, None, None, None

def dataset_prepare(output_path):
    labels = get_kiti_labels()
    with open("kiti_iou_high.jsonl", "rb") as f:
        lines = f.readlines()
    records = []
    max_index = len(lines)
    sampled = [i for i in range(0, max_index, 3)]

    with open("sampled.json", "w", encoding="utf8") as f:
        json.dump(sampled, f)

    for img_idx, line in enumerate(lines):
        if img_idx not in sampled:
            continue
        print(f"Processing image {img_idx+1}/{max_index}")
        data = json.loads(line)
        image = to_png(data.get("index"))
        car_data = data.get("car_iou")
        car_mean = data.get("car_mean")
        car_zero = sum(1 for x in car_data if x == 0.0)
        person_data = data.get("person_iou")
        person_mean = data.get("person_mean")
        person_zero = sum(1 for x in person_data if x == 0.0)
        label,index = labels[data.get("index")]
        index_data = data.get("index")
        print(label)
        if index != data.get("index"):
            print("Tag does not match image index!!!")
            continue
        for i in range(5):
            return_dict = search_one_dim_increment(image, label, car_zero, person_zero, car_mean, person_mean, index_data)
            if return_dict != {}:
                records.append(return_dict)

    with open(output_path, "w", encoding="utf8") as f:
        for index, item in enumerate(records):
            obj = dict(item)
            obj["id"] = index
            line = json.dumps(obj, ensure_ascii=False)
            f.write(line + "\n")

def search_one_dim_increment(
    image,
    label,
    car_zero,
    person_zero,
    car_mean,
    person_mean,
    index_data,
    step_size=0.01,
    rng=None,
):
    if rng is None:
        rng = random.Random()

    max_steps = MAX_STEPS
    step_index = 0

    level_brightness = 0.0
    level_contrast = 0.0
    level_fog = 0.0
    level_blur = 0.0

    dir_brightness = rng.choice((True, False))
    dir_contrast = rng.choice((True, False))
    while step_index <= max_steps:

        group_choice = rng.choice(("A", "B"))

        if group_choice == "A":
            dim_choice = rng.choice(("brightness", "contrast"))
        else:
            dim_choice = rng.choice(("fog", "blur"))

        if dim_choice == "brightness":
            value = step_size
            if level_brightness == 1.0 or level_brightness == -1.0:
                continue
            if not dir_brightness:
                value = float(np.negative(value))
            level_brightness = round(level_brightness + value, 2)
            level_brightness = max(-1.0, min(1.0, level_brightness))

        elif dim_choice == "contrast":
            value = step_size
            if level_contrast == 1.0 or level_contrast ==-1.0:
                continue
            if not dir_contrast:
                value = float(np.negative(value))
            level_contrast = round(level_contrast + value, 2)
            level_contrast = max(-1.0, min(1.0, level_contrast))
            
        elif dim_choice == "fog":
            if level_fog == 1.0:
                continue
            level_fog = round(level_fog + step_size, 2)
            level_fog = max(0.0, min(1.0, level_fog))
            
        elif dim_choice == "blur":
            if level_blur == 1.0:
                continue
            level_blur = round(level_blur + step_size, 2)
            level_blur = max(0.0, min(1.0, level_blur))
            
        perturbed = apply_all(
            image,
            b_brightness=level_brightness,
            c_contrast=level_contrast,
            f_fog=level_fog,
            b_blur=level_blur,
        )

        result, car_iou, people_iou, car_zero_num, person_zero_num = eval_condition(perturbed, label, car_zero, person_zero, car_mean, person_mean)
        if result:
            print("Found suitable condition!")
            return {
                "index": index_data,
                "ori_car_mean": car_mean,
                "car_mean": round(car_iou, 2) if car_iou is not None else None,
                "ori_person_mean": person_mean,
                "person_mean": round(people_iou, 2) if people_iou is not None else None,
                "ori_car_zero": car_zero,
                "car_zero": car_zero_num,
                "ori_person_zero": person_zero,
                "person_zero": person_zero_num,
                "step_index": step_index,
                "brightness": level_brightness,
                "contrast": level_contrast,
                "fog": level_fog,
                "blur": level_blur,
            }
        step_index = step_index + 1

    return {}
