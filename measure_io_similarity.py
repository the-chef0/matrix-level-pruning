from typing import Dict, List

from base_model_utils import BaseModelUtils

import datasets
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from tqdm import tqdm

def is_pruning_candidate(module: Module, module_to_name: Dict[Module, str], excluded_keywords: List[str]) -> bool:
    module_name = module_to_name[module]
    is_matrix = isinstance(module, Linear)
    is_included = not any(excl in module_name for excl in excluded_keywords)
    return is_matrix and is_included


def measure_similarity(model_utils: BaseModelUtils, save_path: str):
    excluded_keywords = ['self_attn', 'lm_head']
    
    hooks = []
    for module in model_utils.base_model.modules():
        if is_pruning_candidate(module, model_utils.module_to_name, excluded_keywords):
            hook = module.register_forward_hook(model_utils.hook_fn)
            hooks.append(hook)

    dataset = datasets.load_dataset("arcee-ai/sec-data-mini", split="train")
    dataloader = DataLoader(dataset['text'], batch_size=2, shuffle=False, num_workers=1)

    print("Measuring input/output similarity on all matrices")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = model_utils.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=128,
                truncation=True).to('cuda')
            model_utils.base_model(**inputs)

    similarity_data = []
    for module_name, similarity_values in model_utils.module_name_to_similarity.items():
        if similarity_values:
            similarity_data.append({
                'module_name': module_name,
                'mean_similarity': np.nanmean(similarity_values),
                'stdev_similarity': np.nanstd(similarity_values)
            })

    similarity_df = pd.DataFrame(similarity_data)
    similarity_df.to_csv(save_path, index=False)

    for hook in hooks:
        hook.remove()
