import json
import os

import lm_eval
from lm_eval.models.huggingface import HFLM
import torch
from transformers import AutoTokenizer

from config.config_protocol import ConfigProtocol
from infra.utils.model_utils import ModelUtils

def evaluate_pruned(cfg: ConfigProtocol, model_utils: ModelUtils):
    # Initialize the model and tokenizer
    if model_utils:
        model = model_utils.model
        tokenizer = model_utils.tokenizer
    else:
        model = torch.load(os.path.join(cfg.PRUNED_MODEL_SAVE_DIR, "model.pth"), weights_only=False).to(cfg.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(cfg.PRUNED_MODEL_SAVE_DIR)

    name = os.path.basename(cfg.PRUNED_MODEL_SAVE_DIR)

    llm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda",
        batch_size='auto'
    )

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=llm,
        model_args=name,
        tasks=["hellaswag", "piqa", "wsc273", "coqa", "race", "mmlu", "cmmlu", "boolq"], # missing CMNLI, CHID, XSum, C3
        device="cuda",
    )

    print(results)

    with open(cfg.EVAL_RESULTS_PATH, "w") as f:
        json.dump(results['results'], f)
