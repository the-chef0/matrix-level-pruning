from utils.model_utils import ModelUtils

import json
import os

import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from transformers import AutoTokenizer

def evaluate_pruned(model_utils: ModelUtils, pruned_model_save_dir: str, eval_result_path: str):
    # Initialize the model and tokenizer
    if model_utils:
        model = model_utils.model
        tokenizer = model_utils.tokenizer
    else:
        model = torch.load(os.path.join(pruned_model_save_dir, "model.pth"), weights_only=False).cuda()
        tokenizer = AutoTokenizer.from_pretrained(pruned_model_save_dir)

    name = os.path.basename(pruned_model_save_dir)

    llm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda"
    )

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=llm,
        model_args=name,
        tasks=["hellaswag"],
        batch_size=16,
        device="cuda",
    )

    with open(eval_result_path, "w") as f:
        json.dump(results['results'], f)
