# train_qwen3_vl_sft.py
# complete supervised fine tuning script for Qwen three VL eight B Instruct
# this script uses Hugging Face transformers Trainer

import os
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tools import *
from typing import List, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

class JsonlVisionDataset(Dataset):
    
    def __init__(self, jsonl_path):
        self.items = []

        with open(jsonl_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                image_path = to_png(obj["index"])

                ori_car_mean = obj.get("ori_car_mean", None)
                ori_person_mean = obj.get("ori_person_mean", None)
                car_mean = obj.get("car_mean", None)
                person_mean = obj.get("person_mean", None)
                car_zeroed = obj.get("car_zero", None)
                car_zeroed_ori = obj.get("ori_car_zero", None)
                person_zeroed = obj.get("person_zero", None)
                person_zeroed_ori = obj.get("ori_person_zero", None)
                

                brightness = obj["brightness"]
                contrast = obj["contrast"]
                fog = obj["fog"]
                blur = obj["blur"]

                self.items.append(
                    {
                        "image_path": image_path,
                        "ori_car_mean": ori_car_mean,
                        "ori_person_mean": ori_person_mean,
                        "car_mean": car_mean,
                        "person_mean": person_mean,
                        "brightness": brightness * 100,
                        "contrast": contrast * 100,
                        "fog": fog * 100,
                        "blur": blur * 100,
                        "car_zero": car_zeroed,
                        "ori_car_zero": car_zeroed_ori,
                        "person_zero": person_zeroed,
                        "ori_person_zero": person_zeroed_ori,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    
    def __items__(self):
        return self.items


class QwenThreeVLDataCollator:

    processor: object
    max_length: int = 2048
    car_thr: float = 0.7
    person_thr: float = 0.5
    
    def __init__(self, processor, system_prompt, max_length=2048, car_thr=0.7, person_thr=0.5):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.car_thr = car_thr
        self.person_thr = person_thr
        self.ignore_index = self.tokenizer.pad_token_id

    def _fmt_mean(self, value):
        if value is None:
            return "No Data"
        return f"{value:.3f}"

    def _find_last_subsequence_start(self, seq: List[int], subseq: List[int]):
        if not subseq:
            return None
        n = len(seq)
        m = len(subseq)
        if m > n:
            return None
        last = None
        for start in range(n):
            end = start + m
            if end > n:
                break
            if seq[start:end] == subseq:
                last = start
        return last
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_messages = []
        answer_texts: List[str] = []
        regression_values = []

        for sample in features:
            image_path = sample["image_path"]
            base_car_mean = sample["ori_car_mean"]
            base_person_mean = sample["ori_person_mean"]
            car_mean_after = sample["car_mean"]
            person_mean_after = sample["person_mean"]
            car_zeroed = sample["car_zero"]
            car_zeroed_ori = sample["ori_car_zero"]
            person_zeroed = sample["person_zero"]
            person_zeroed_ori = sample["ori_person_zero"]

            brightness = sample["brightness"]
            contrast = sample["contrast"]
            fog = sample["fog"]
            blur = sample["blur"]

            base_car_text = self._fmt_mean(base_car_mean)
            base_person_text = self._fmt_mean(base_person_mean)

            system_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                    }
                ],
            }
            if base_car_text == "No Data":
                car_text = "There is no detected cars in the image."
            else:

                car_text = f"The average IoU for cars is {base_car_text}."

            if base_person_text == "No Data":
                person_text = "There is no detected person in the image."
            else:
                person_text = f"The average IoU for people is {base_person_text}."
            
            user_text = (
                "This is a forward-facing driving image. "
                f"{car_text} "
                f"{person_text} "
                f"Think step by step to design a set of perturbations to reduce the average IOU for cars below 0.7, the average IOU for people below 0.5, or to increase the number of missed detections. "
                "The perturbation parameters are four normalized values: brightness, contrast, fog, and blur. "
                "The brightness and contrast values range from -100 to 100. The fog and blur values range from 0 to 100. "
                "Output only one JSON object."
            )

            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            }

            answer_obj = {
                "brightness": brightness,
                "contrast": contrast,
                "fog": fog,
                "blur": blur
            }

            answer_json = json.dumps(answer_obj, ensure_ascii=False)

            assistant_message = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer_json,
                    }
                ],
            }

            messages = [system_message, user_message, assistant_message]
            batch_messages.append(messages)
            answer_texts.append(answer_json)

            regression_values.append(
                [
                    float(brightness),
                    float(contrast),
                    float(fog),
                    float(blur),
                ]
            )

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=False,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        labels = torch.full_like(input_ids, fill_value=self.ignore_index)

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            ids_row = input_ids[i].tolist()

            ans_ids = self.tokenizer(
                answer_texts[i],
                add_special_tokens=False,
            )["input_ids"]

            start = self._find_last_subsequence_start(ids_row, ans_ids)
            if start is None:
                print("Cannot find answer subsequence in input ids!")
                # 这一条找不到答案子序列 就不提供监督
                continue

            end = start + len(ans_ids)
            if end > input_ids.shape[1]:
                end = input_ids.shape[1]

            labels[i, start:end] = input_ids[i, start:end]

        labels[attention_mask == 0] = self.ignore_index

        model_inputs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "regression_labels": torch.tensor(regression_values, dtype=torch.float32),
        }

        for key, value in inputs.items():
            if key in model_inputs:
                continue
            model_inputs[key] = value

        return model_inputs

class QwenThreeVLTrainer(Trainer):
    regression_loss_weight: float = 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        regression_labels = inputs.pop("regression_labels", None)

        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss

        if regression_labels is None:
            if return_outputs:
                return lm_loss, outputs
            return lm_loss

        hidden_states = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[len(outputs.hidden_states) - 1]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Model outputs do not contain hidden states for regression pooling")

        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)

        if attention_mask is None:
            pooled_fp32 = hidden_states.float().mean(dim=1)
        else:
            attn_bool = attention_mask.to(dtype=torch.bool)

            if labels is not None:
                ignore_index = torch.nn.CrossEntropyLoss().ignore_index
                label_bool = labels.ne(ignore_index) & attn_bool
                label_counts = label_bool.sum(dim=1)
                use_label = label_counts.gt(0)
                pool_bool = torch.where(use_label.unsqueeze(1), label_bool, attn_bool)
            else:
                pool_bool = attn_bool

            pool_mask = pool_bool.unsqueeze(2).to(dtype=torch.float32)
            summed = (hidden_states.float() * pool_mask).sum(dim=1)
            lengths = pool_mask.sum(dim=1).clamp(min=torch.finfo(torch.float32).eps)
            pooled_fp32 = summed / lengths

        pooled = pooled_fp32.to(dtype=hidden_states.dtype, device=hidden_states.device)

        if not hasattr(model, "regression_head") or model.regression_head is None:
            raise ValueError("model.regression_head is missing")

        if next(model.regression_head.parameters()).device != hidden_states.device:
            model.regression_head = model.regression_head.to(hidden_states.device)

        preds = model.regression_head(pooled)

        regression_labels = regression_labels.to(device=preds.device, dtype=torch.float32)
        mse_loss = F.mse_loss(preds.float(), regression_labels)

        total_loss = lm_loss + (self.regression_loss_weight * mse_loss)

        if return_outputs:
            return total_loss, outputs
        return total_loss
    
def start():
    root = "./"
    train_dataset_prepare("./kiti_perturbed.jsonl")
    train_jsonl = os.path.join(root, "kiti_perturbed_train.jsonl")
    eval_jsonl = os.path.join(root, "kiti_perturbed_eval.jsonl")

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.output_hidden_states = True

    if not hasattr(model, "regression_head"):
        embed = model.get_input_embeddings()
        hidden_size = getattr(embed, "embedding_dim", None)
        if hidden_size is None:
            hidden_size = embed.weight.shape[1]
        model.regression_head = nn.Linear(hidden_size, 4)

    train_dataset = JsonlVisionDataset(train_jsonl)
    print(train_dataset)
    eval_dataset = JsonlVisionDataset(eval_jsonl)

    SYSTEM_PROMPT = """You are an agent for autonomous driving perception.
    Input: One driving image.
    Objective: Propose the minimal perturbations that are most likely to push the car IoU below 0.7 or the person IoU below 0.5, using the factors brightness, contrast, fog, and blur. It is also acceptable if these perturbations result in more missed detections.
    Range: The factors brightness and contrast is in the range of -100 to 100; the factors blur and fog is in the range of 0 to 100.
    Cost: The sum of the absolute values of the four perturbation factors.
    Output: Return JSON only with this schema and replace ** with concrete numbers:
    {
        "brightness": **,
        "contrast": **,
        "fog": **,
        "blur": **,
    }"""

    data_collator = QwenThreeVLDataCollator(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        max_length=2048,
    )

    learning_rate_value = math.pow(10.0, 0.0) / math.pow(10.0, 5.0)

    training_args = TrainingArguments(
        output_dir="qwen3_output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        learning_rate=1e-3,
        warmup_ratio=0.03,
        gradient_accumulation_steps=4,
        logging_steps=20,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        remove_unused_columns=False,
    )

    # six create trainer
    trainer = QwenThreeVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # seven start training
    trainer.train()

    # eight save final model and processor
    trainer.save_model("qwen3_output")
    processor.save_pretrained("qwen3_output")

if __name__ == "__main__":
    start()
