import json
import math
import numpy as np
import os
import random
import logging
from ultralytics.utils import LOGGER
from ultralytics import YOLO
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm import LLM, SamplingParams
from perturbation import apply_brightness, apply_contrast, apply_fog, apply_blur
from qwen_vl_utils import process_vision_info
from getLabels import get_kiti_labels
from json_repair import repair_json
import cv2

GPU_PER = 0.65
LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

MAPPING_DATASET = {
    0: "image_kiti",
    1: "image_bdd"
}

MAPPING = {
    "2": 2,
    "3": 2,
    "5": 2,
    "6": 2,
    "7": 2,
    "0": 0
}

SYSTEM_PROMPT = """You are an agent for autonomous driving perception.
    Input: one driving image.
    Objective: propose the smallest perturbations that are most likely to push the car IoU below 0.7 or the person IoU below 0.5, using the keys brightness, contrast, fog, and blur. It is also acceptable if these perturbations result in more missed detections.
    Requirment: The keys brightness and contrast is in the range of -100 to 100; the keys blur and fog is in the range of 0 to 100.
    Output: Return JSON only with this schema and replace ** with concrete numbers:
    {
        "brightness": **,
        "contrast": **,
        "fog": **,
        "blur": **,
    }"""

def delete_zero(arr):
    cleaned = [[x for x in row if x != 0.0] for row in arr]

    first_row = arr[0]
    zero_count = sum(1 for x in first_row if x == 0.0)

    return cleaned, zero_count


def compute_mean(car_formatted, person_formatted):
    def mean_ignore_zeros(values):
        non_zero_values = [v for v in values if v != 0.0]
        if non_zero_values:
            return sum(non_zero_values) / len(non_zero_values)
        else:
            return None

    car_mean = mean_ignore_zeros(car_formatted)
    people_mean = mean_ignore_zeros(person_formatted)

    return car_mean, people_mean

def save_jsonl(path, all_car_image_ious, all_person_image_ious):
    with open(path, "w", encoding="utf8") as f:
        count = 0
        for index, (car_row, person_row) in enumerate(zip(all_car_image_ious, all_person_image_ious)):
            # 車
            car_formatted = [float(f"{v:.4f}") for v in car_row]

            # 人
            person_formatted = [float(f"{v:.4f}") for v in person_row]
            car_avg, person_avg = compute_mean(car_formatted, person_formatted)

            row_dict = {
                "id": count,
                "index": index,
                "car_iou": car_formatted,
                "car_mean": float(f"{car_avg:.4f}") if car_avg is not None else None,
                "person_iou": person_formatted,
                "person_mean": float(f"{person_avg:.4f}") if person_avg is not None else None,
            }

            json.dump(row_dict, f, ensure_ascii=False)
            f.write("\n")
            count += 1

def too_many_zeros(values):
    if not isinstance(values, list):
        return False
    if len(values) == 0:
        return False
    zero_count = 0
    non_zero_count = 0
    for v in values:
        if not isinstance(v, (int, float)):
            continue
        if v == 0.0:
            zero_count += 1
        else:
            non_zero_count += 1
    return zero_count > non_zero_count

def extract_good_images(in_path, out_path):
    selected = []

    with open(in_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            car_mean = obj.get("car_mean")
            person_mean = obj.get("person_mean")
            car_iou_list = obj.get("car_iou") or []
            person_iou_list = obj.get("person_iou") or []

            if isinstance(car_mean, (int, float)):
                car_ok = car_mean > 0.7
            else:
                car_ok = True

            # 人類條件
            if isinstance(person_mean, (int, float)):
                person_ok = person_mean > 0.5
            else:
                person_ok = True

            car_zero_bad = too_many_zeros(car_iou_list)
            person_zero_bad = too_many_zeros(person_iou_list)

            if car_zero_bad or person_zero_bad:
                continue
            if car_ok and person_ok:
                selected.append(obj)

    with open(out_path, "w", encoding="utf8") as f:
        for new_id, obj in enumerate(selected):
            new_obj = dict(obj)
            new_obj["id"] = new_id
            json.dump(new_obj, f, ensure_ascii=False)
            f.write("\n")

def to_png(n, width=6, ext="png", dataset=0):
    return f"{MAPPING_DATASET[dataset]}/{int(n):0{width}d}.{ext}"

def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,   # 四张 GPU
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_PER      # FP16 / BF16 自动选择
    )
    return llm, processor

def fmt_mean(value):
    if value is None:
        return "No Data"
    return f"{value:.3f}"

def load_prompt(processor, path):
    requests = []
    information = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            image_path = to_png(obj.get("index", 0))

            base_car_text = fmt_mean(obj.get("car_mean"))
            base_person_text = fmt_mean(obj.get("person_mean"))

            if base_car_text == "No Data":
                car_text = "There is no detected vehicle in the image."
            else:

                car_text = f"The average IoU for vehicle is {base_car_text}."

            if base_person_text == "No Data":
                person_text = "There is no detected person in the image."
            else:
                person_text = f"The average IoU for people is {base_person_text}."

            user_text = (
                "This is a forward-facing driving image. "
                f"{car_text} "
                f"{person_text} "
                f"Think step by step to design a set of perturbations to reduce the average IOU for vehicles below 0.7, the average IOU for people below 0.5, or to increase the number of missed detections. "
                "The perturbation parameters are four normalized values: brightness, contrast, fog, and blur. "
                "The brightness and contrast values range from -100 to 100. The fog and blur values range from 0 to 100. "
                "Output only one JSON object."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 从同一份 messages 提取图像特征并封装为 vLLM 请求
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)
            mm = {}
            if image_inputs is not None:
                mm["image"] = image_inputs
            if video_inputs is not None:
                mm["video"] = video_inputs

            req = {
                "prompt": prompt,
                "multi_modal_data": mm,
                "mm_processor_kwargs": video_kwargs
            }
            requests.append(req)

            information.append({
                "id": obj.get("id", 0),
                "index": obj.get("index", 0),
                "mean": obj.get("mean", None),
                "car_iou": obj.get("car_iou", []),
                "person_iou": obj.get("person_iou", [])
            })

    return requests, information

def run_model(requests, information, batch_size, model, out_path = "kiti_outputs.jsonl"):
    predictions = []
    counter = 1
    for i in range(0, len(requests), batch_size):
        batch_prompts = requests[i:i+batch_size]
        batch_predictions = run_batch(batch_prompts, model)
        predictions.extend(batch_predictions)

        if (i // 200) == counter:
            print(f"\033[1;32m{i}\033[0m instances generated successfully")
            counter += 1

    print("Starting to compute...")
    save_outputs(predictions, information, out_path)
    return predictions

def run_batch(batch_prompts, model):
    predictions = []

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        stop_token_ids=[]
    )

    outputs = model.generate(batch_prompts, sampling_params)
    # Decode input and output to strings

    predictions = [output.outputs[0].text.strip() for output in outputs]

    return predictions

def save_outputs(predictions, information, out_path):
    with open(out_path, "w", encoding="utf8") as f:
        for pred, info in zip(predictions, information):
            obj = {
                "id": info["id"],
                "index": info["index"],
                "mean": info["mean"],
                "car_iou": info["car_iou"],
                "person_iou": info["person_iou"],
                "output": pred
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

# 计算 IOU
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def load_compute(pred_boxes, gt_boxes, MIN_COUNT, MAX_COUNT):
    all_car_image_ious = []
    all_person_image_ious = []

    for pb_list, gb_list in zip(pred_boxes, gt_boxes[MIN_COUNT: MAX_COUNT]):
        car_image_ious = []
        person_image_ious = []
        for gb in gb_list:
            gb_box = [gb["xmin"], gb["ymin"], gb["xmax"], gb["ymax"]]

            max_iou = 0.0
            for pb in pb_list:
                pb_box = [pb["xmin"], pb["ymin"], pb["xmax"], pb["ymax"]]
                iou = compute_iou(pb_box, gb_box)
                if gb["type"] != pb["type"]:
                    continue
                if iou > max_iou:
                    max_iou = iou
            if gb["type"] == 2:
                car_image_ious.append(max_iou)
            elif gb["type"] == 0:
                person_image_ious.append(max_iou)
            else:
                pass

        all_car_image_ious.append(car_image_ious)
        all_person_image_ious.append(person_image_ious)

    return all_car_image_ious, all_person_image_ious

def train_dataset_prepare(root):

    dir_name = os.path.dirname(root)
    if dir_name == "":
        dir_name = "."

    base_name = os.path.basename(root)
    name_without_ext, ext = os.path.splitext(base_name)

    train_path = os.path.join(dir_name, f"{name_without_ext}_train{ext}")
    eval_path = os.path.join(dir_name, f"{name_without_ext}_eval{ext}")

    with open(root, "r", encoding="utf8") as f:
        lines = f.readlines()

    total = len(lines)
    split_index = int(total * 0.9)

    with open(train_path, "w", encoding="utf8") as f_train:
        for line in lines[:split_index]:
            f_train.write(line)

    with open(eval_path, "w", encoding="utf8") as f_eval:
        for line in lines[split_index:]:
            f_eval.write(line)

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
    print(f"Applying perturbations: brightness={b_brightness}, contrast={c_contrast}, fog={f_fog}, blur={b_blur}")
    img = cv2.imread(image)
    img = img.astype(np.uint8)
    img = apply_brightness(img, b_brightness, factor_min=factor_min, factor_max=factor_max)
    img = apply_contrast(img, c_contrast, alpha_min=alpha_min, alpha_max=alpha_max)
    img = apply_fog(img, f_fog, fog_max=fog_max)
    img = apply_blur(img, b_blur, sigma_max=sigma_max)
    return img

def eval_condition(image, label, car_zero, person_zero):
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
    car_mean, people_mean = compute_mean(car_image_iou[0], person_image_iou[0])
    if (car_mean is not None and car_mean < 0.7) or (people_mean is not None and people_mean < 0.5):
        return True, car_mean, people_mean, car_zero_num, person_zero_num
    if car_zero < car_zero_num or person_zero < person_zero_num:
        return True, car_mean, people_mean, car_zero_num, person_zero_num
    return False, None, None, None, None

def eval_result(path, output_path):
    labels = get_kiti_labels()
    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
    results = []
    count = 0
    sucess_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        output = obj.get("output", "")
        index = obj.get("index", 0)
        output = repair_json(output)
        result = json.loads(output)
        try:
            brightness = result.get("brightness", 0.0)
            contrast = result.get("contrast", 0.0)
            fog = result.get("fog", 0.0)
            blur = result.get("blur", 0.0)
            car_iou = obj.get("car_iou", [])
            people_iou = obj.get("person_iou", [])
        except Exception as e:
            print("error when processing item:", e)
            continue
        if type(fog) is str:
            fog = 0.0
        if type(blur) is str:
            blur = 0.0
        if type(brightness) is str:
            brightness = 0.0
        if type(contrast) is str:
            contrast = 0.0

        # 最原始
        # contrast = max(-100.0, min(100.0, contrast))
        # brightness = max(-100.0, min(100.0, brightness))
        # fog = max(0.0, min(100.0, fog))
        # blur = max(0.0, min(100.0, blur))

        # bias_contrast = -10.0 if contrast < 0.0 else 10.0
        # bias_brightness = -10.0 if brightness < 0.0 else 10.0
        # contrast = max(-100.0, min(100.0, contrast + bias_contrast))
        # brightness = max(-100.0, min(100.0, brightness + bias_brightness))
        # fog = max(0.0, min(100.0, fog + 20))
        # blur = max(0.0, min(100.0, blur + 20))

        # bias_contrast = 5.0 if contrast < 0.0 else -5.0
        # bias_brightness = 5.0 if brightness < 0.0 else -5.0
        # contrast = max(-100.0, min(100.0, contrast + bias_contrast))
        # brightness = max(-100.0, min(100.0, brightness + bias_brightness))
        # fog = max(0.0, min(100.0, fog - 5)) 
        # blur = max(0.0, min(100.0, blur - 5)) 

        # contrast = max(-100.0, min(100.0, random.randint(-100, 100)))
        # brightness = max(-100.0, min(100.0, random.randint(-100, 100)))
        # fog = max(0.0, min(100.0, random.randint(0, 100)))
        # blur = max(0.0, min(100.0, random.randint(0, 100)))

        # 恒定数值
        contrast = max(-100.0, min(100.0, 50))
        brightness = max(-100.0, min(100.0, 50))
        fog = max(0.0, min(100.0, 50))
        blur = max(0.0, min(100.0, 50))

        cost = min(100.0, abs(brightness)) + min(100.0, abs(contrast)) + min(100.0, abs(fog)) + min(100.0, abs(blur))

        person_zero = sum(1 for x in people_iou if x == 0.0)
        car_zero = sum(1 for x in car_iou if x == 0.0)

        label,index = labels[index]
        perturbed = apply_all(to_png(index), b_brightness=(round(brightness/100,2)), c_contrast=(round(contrast/100,2)), f_fog=(round(fog/100,2)), b_blur=(round(blur/100,2)))

        result, car_iou, people_iou, car_zero_num, person_zero_num = eval_condition(perturbed, label, car_zero, person_zero)
        results.append({
            "id": count,
            "index": index,
            "brightness": brightness,
            "contrast": contrast,
            "fog": fog,
            "blur": blur,
            "cost": cost,
            "car_iou": car_iou,
            "people_iou": people_iou,
            "car_zero": car_zero_num,
            "car_zero_before": car_zero,
            "person_zero": person_zero_num,
            "person_zero_before": person_zero,
            "success": result
        })
        if result:
            sucess_count += 1
        count += 1
    with open(output_path, "w", encoding="utf8") as f:
        for obj in results:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
        
    print("Total:", count, "Success:", sucess_count, "Success Rate:", f"{(sucess_count/count)*100:.2f}%", "Average Cost:", f"{sum(r['cost'] for r in results)/count:.2f}")

def clean_trainset(path_in, path_out, sampled_path="sampled.json"):
    # 读需要剔除的 index 列表
    with open(sampled_path, "r", encoding="utf8") as f:
        sampled = json.load(f)
    sampled_set = set(sampled)

    count = 0
    with open(path_in, "r", encoding="utf8") as fin, open(path_out, "w", encoding="utf8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            idx = obj.get("id")
            # 在 sampled 里的样本剔除
            if idx in sampled_set:
                continue

            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")
            count += 1

    print("Cleaned training set size:", count)
