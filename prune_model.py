from dataclasses import dataclass
import os
from typing import Callable, List, Tuple, Union

from base_model_utils import BaseModelUtils

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter
import torch_pruning as tp
from torch_pruning import DependencyGraph, Group
from torch_pruning.pruner.function import LayernormPruner
from transformers.models.llama.modeling_llama import LlamaRMSNorm

@dataclass
class DepSearchDirection:
    backward: Callable = tp.prune_linear_in_channels
    forward: Callable = tp.prune_linear_out_channels
    not_needed: Callable = None

@dataclass
class CompressionAxis:
    row: Callable = tp.prune_linear_out_channels
    col: Callable = tp.prune_linear_in_channels

def extract_logits(output: Tensor) -> Tensor:
    return output.logits

def get_dep_search_direction(module: Module) -> DepSearchDirection:
    assert isinstance(module, Linear), "Only linear layers are supported for now"

    if module.in_features > module.out_features:
        return DepSearchDirection.backward
    elif module.in_features < module.out_features:
        return DepSearchDirection.forward
    else:
        return DepSearchDirection.not_needed
    
def get_deps(module: Module, dep_graph: DependencyGraph) -> List[Tuple[Module, CompressionAxis]]:
    assert isinstance(module, Linear), "Only linear layers are supported for now"

    search_direction = get_dep_search_direction(module)

    if search_direction == DepSearchDirection.not_needed:
        return []
    else:
        dependency_group = dep_graph.get_pruning_group(module, search_direction, idxs=[1])
        return get_dep_compression_targets(dependency_group)

def get_dep_compression_targets(dependency_group: Group) -> List[Tuple[Module, CompressionAxis]]:
    compression_pairs = []

    for item in dependency_group[1:]:
        curr_dep = item.dep
        if isinstance(curr_dep.target.module, Linear):
            compression_pairs.append((curr_dep.target.module, curr_dep.handler))

    return compression_pairs

def replace_module_by_name(model_utils: BaseModelUtils, module_name, new_module):
    # Split the module name into parts
    parts = module_name.split('.')
    
    # Get the parent module (everything except the last part)
    parent = model_utils.base_model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    # Replace the module
    setattr(parent, parts[-1], new_module)

def compress_module_matrix(model_utils: BaseModelUtils, module: Module, axis: CompressionAxis, target_dim: int):
    U, S, Vh = torch.linalg.svd(module.weight, full_matrices=False)
          
    if axis == CompressionAxis.row:
        S = torch.diag(S[:target_dim])
        compressed_weight = torch.matmul(S, Vh)
    elif axis == CompressionAxis.col:
        S = torch.diag(S[:target_dim])
        compressed_weight = torch.matmul(U, S)
    
    compressed_module = Linear(compressed_weight.shape[1], compressed_weight.shape[0], bias=False)
    compressed_module.weight = Parameter(compressed_weight)
    compressed_module_name = model_utils.module_to_name[module]
    replace_module_by_name(model_utils, compressed_module_name, compressed_module)

def prune_module_matrix(model_utils: BaseModelUtils, module_name: str, dep_graph: DependencyGraph):
    module = model_utils.name_to_module[module_name]
    target_dim = np.min((module.in_features, module.out_features))
    deps = get_deps(module, dep_graph)

    for dep_module, axis in deps:
        print(f"Compressing {model_utils.module_to_name[dep_module]} to {target_dim}")
        compress_module_matrix(model_utils, dep_module, axis, target_dim)

    replace_module_by_name(model_utils, module_name, torch.nn.Identity())

def prune(model_utils: BaseModelUtils, similarity_save_path: str, matrices_to_prune: Union[int, List[str]], pruned_model_save_dir: str):

    if isinstance(matrices_to_prune, int):
        similarity_df = pd.read_csv(similarity_save_path)
        similarity_df = similarity_df.sort_values(by='mean_similarity', ascending=False)
        # TODO: Make a constants file
        modules_to_prune = similarity_df.iloc[:matrices_to_prune]['module_name'].tolist()

    if isinstance(matrices_to_prune, list):
        modules_to_prune = matrices_to_prune

    dep_graph = tp.DependencyGraph().build_dependency(
        model=model_utils.base_model,
        example_inputs=model_utils.base_model.dummy_inputs['input_ids'].to('cuda'),
        output_transform=extract_logits,
        customized_pruners={LlamaRMSNorm: LayernormPruner()}
    )
    print("Dependency graph built")

    for module_name in modules_to_prune:
        print(f"Pruning {module_name}")
        prune_module_matrix(model_utils, module_name, dep_graph)

    pruned_model = model_utils.base_model

    model_utils.tokenizer.save_pretrained(pruned_model_save_dir)
    torch.save(pruned_model, os.path.join(pruned_model_save_dir, "model.pth"))
    print("Pruned model saved")
