import torch
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def strip_regression_head():
    src_dir = "qwen3_output"          # 你训练完保存的目录
    dst_dir = "qwen3_output_vllm"     # 给 vLLM 用的新目录

    for d in (src_dir, dst_dir): os.makedirs(d, exist_ok=True)
    
    print("loading model from", src_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        src_dir,
        torch_dtype=torch.float16,
    )

    # 打印一下看一眼
    print("has regression_head attribute before strip:", hasattr(model, "regression_head"))

    # 如果真的有 就删掉这个属性
    if hasattr(model, "regression_head"):
        delattr(model, "regression_head")

    print("has regression_head attribute after strip:", hasattr(model, "regression_head"))

    # 再检查一下 state dict 里有没有 regression_head 相关 key
    state_dict = model.state_dict()
    bad_keys = [k for k in state_dict.keys() if "regression_head" in k]
    print("number of regression_head keys in state dict:", len(bad_keys))
    if bad_keys:
        print("example bad key:", bad_keys[0])

    # 保存给 vLLM 用的精简版
    model.save_pretrained(dst_dir)
    print("saved vLLM friendly checkpoint to", dst_dir)

    # processor 顺手拷一份 方便一致使用
    processor = AutoProcessor.from_pretrained(src_dir)
    processor.save_pretrained(dst_dir)
    print("copied processor to", dst_dir)