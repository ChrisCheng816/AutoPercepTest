# from ultralytics import YOLO
# from getLabels import get_kiti_labels
# from get_images_res import kiti_images_res
# import cv2

# model = YOLO('yolov8n.pt')

# # 计算 IOU
# def compute_iou(box1, box2):
#     xA = max(box1[0], box2[0])
#     yA = max(box1[1], box2[1])
#     xB = min(box1[2], box2[2])
#     yB = min(box1[3], box2[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
#     box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
#     iou = interArea / float(box1Area + box2Area - interArea)
#     return iou

# def load_compute(pred_boxes, gt_boxes):
#     """
#     pred_boxes: 每张图的预测框列表 形如 [[ [x1,y1,x2,y2], ... ], ...]
#     gt_boxes: 每张图的 gt 列表 形如 [[ {"xmin":..,"ymin":..,"xmax":..,"ymax":..}, ... ], ...]
#     返回每张图中每个 gt 对应的最大 IoU
#     """
#     all_image_max_ious = []

#     for pb_list, gb_list in zip(pred_boxes, gt_boxes):
#         image_max_ious = []
#         for gb in gb_list:
#             gb_box = [gb["xmin"], gb["ymin"], gb["xmax"], gb["ymax"]]

#             max_iou = 0.0
#             for pb in pb_list:
#                 iou = compute_iou(pb, gb_box)
#                 if iou > max_iou:
#                     max_iou = iou

#             image_max_ious.append(max_iou)
#         all_image_max_ious.append(image_max_ious)

#     return all_image_max_ious

# if __name__ == "__main__":
#     results = model("test.png")
#     arr = results[0].boxes.xyxy.tolist()
#     label = [[{"xmin":712.40,"ymin":143.00,"xmax":810.73,"ymax":307.92}]]
#     iou = load_compute([arr], label)
#     print(iou)


# def increase_blur(img, base_kernel=5, percent=20):
#     """让图像模糊度提高指定百分比"""
#     new_kernel = int(base_kernel * (1 + percent / 100))
#     if new_kernel % 2 == 0:
#         new_kernel += 1
#     blurred = cv2.GaussianBlur(img, (new_kernel, new_kernel), 0)
#     return blurred

# # 用法示例
# img = cv2.imread("test.png")
# out = increase_blur(img, base_kernel=5, percent=200)
# cv2.imwrite("test_blurred.png", out)

import json

count = 0
missed = 0
with open("original_outputs.jsonl", "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        result = data.get("success")
        car_zero = data.get("car_zero")
        car_zero_before= data.get("car_zero_before")
        person_zero = data.get("person_zero")
        person_zero_before = data.get("person_zero_before")
        if result == True:
            count = count + 1
            if (car_zero is not None and car_zero > car_zero_before) or (person_zero is not None and person_zero > person_zero_before):
                missed = missed + 1

print("count=",count)
print("missed=",missed)


