from ultralytics import YOLO
from getLabels import get_kiti_labels
from get_images_res import kiti_images_res
from tools import *
from finetuning import start
from strip_regression_head import strip_regression_head
from prepare_dataset import dataset_prepare
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MIN_COUNT = 0
MAX_COUNT = 7481
KITI_PATH = "kiti_iou.jsonl"
KITI_HIGH_PATH = "kiti_iou_high.jsonl"
KITI_EVAL_HITH_PATH = "kiti_high_eval.jsonl"

DATASET_KITI = 0
DATASET_BDD = 1
BATCH_SIZE = 16

def deal_kiti_iou():
    all_labels = get_kiti_labels()
    gt_boxes = [labels for labels, index in all_labels]
    image_dict = kiti_images_res(MIN_COUNT, MAX_COUNT)
    print("Loaded", len(image_dict), "images and", len(gt_boxes), "labels.")
    all_car_image_ious, all_person_image_ious = load_compute(image_dict, gt_boxes, MIN_COUNT, MAX_COUNT)
    save_jsonl(KITI_PATH,all_car_image_ious, all_person_image_ious)

def generate_outputs(path, out_path):
    llm, processor = load_model(path)
    requests, information = load_prompt(processor, KITI_EVAL_HITH_PATH)
    run_model(requests, information, BATCH_SIZE, llm, out_path)

if __name__ == "__main__":
    # deal_kiti_iou()
    # extract_good_images(KITI_PATH, KITI_HIGH_PATH)

    # dataset_prepare("kiti_perturbed.jsonl")

    # clean_trainset(KITI_HIGH_PATH, KITI_EVAL_HITH_PATH)

    # start()

    # strip_regression_head()

    #--------------------------------------------------------------------
    # generate_outputs("qwen3_output_vllm", "answer_finetuned.jsonl")
    # eval_result("answer_finetuned.jsonl", "outputs_finetuned.jsonl")

    generate_outputs("Qwen/Qwen3-VL-8B-Instruct", "answer_brute_50.jsonl")
    eval_result("answer_brute_50.jsonl", "outputs_brute_50.jsonl")