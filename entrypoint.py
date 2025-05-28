import argparse
import os
import gc

from eval_pruned import evaluate_pruned
from importances_and_groups import collect_groups
from prune_model import prune
from utils.base_model_utils import BaseModelUtils

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix-level pruning pipeline')
    
    # Importance measurement arguments
    parser.add_argument('--base_model_id', type=str, default=None,
                        help='HuggingFace model ID of the base model')
    parser.add_argument('--importances_save_path', type=str, default='./similarity.csv',
                        help='Path where to save the measured similarities')

    # Pruning arguments
    parser.add_argument('--pruning_iterations', type=str, nargs='+', default='1',
                        help='Number of times to rank group importances and prune the least important group')
    parser.add_argument('--pruned_model_save_dir', type=str, default=None,
                        help='Directory where to save the pruned model')

    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate the pruned model')
    parser.add_argument('--eval_result_path', type=str, default='./eval-result.json',
                        help='Path where to save the evaluation results')

    args = parser.parse_args()
    args.pruning_iterations = int(args.pruning_iterations[0])

    return args

args = parse_args()

base_model_utils = None

if args.base_model_id:
    base_model_utils = BaseModelUtils(args.base_model_id)
    print("Base model loaded")

assert args.pruning_iterations >= 0
for i in range(args.pruning_iterations):
    print(f"Iteration {i + 1}")
    # TODO: Make this more efficient
    # Every iteration after the first re-collects groups for the entire model,
    # but we know which modules changed in each pruning iteration.
    # It should suffice to just re-collect groups for the modules affected by pruning.

    print("(Re)building module - name mappings")
    base_model_utils.build_module_name_mappings()
    print("(Re)building dependency graph")
    base_model_utils.build_dependency_graph()

    importances_and_groups = collect_groups(
        base_model_utils,
        iteration=i,
        save_path=args.importances_save_path
    )
    _, group_to_prune = importances_and_groups.pop(0)
    prune(base_model_utils, group_to_prune)

pruned_model = base_model_utils.base_model
# print(pruned_model)
base_model_utils.tokenizer.save_pretrained(args.pruned_model_save_dir)
torch.save(pruned_model, os.path.join(args.pruned_model_save_dir, "model.pth"))
print(f"Saved pruned model to {args.pruned_model_save_dir}")

if args.evaluate:
    assert args.pruned_model_save_dir is not None
    assert args.eval_result_path is not None

    if base_model_utils is not None:
        # Remove the base model from memory before evaluate_pruned loads the pruned one. 
        # This might not be necessary on other GPUs but my poor RTX 4060 does not have
        # the VRAM to hold both.
        del base_model_utils.base_model
        del base_model_utils.tokenizer
        del base_model_utils
        gc.collect()
        torch.cuda.empty_cache()
    
    evaluate_pruned(
        pruned_model_save_dir=args.pruned_model_save_dir,
        eval_result_path=args.eval_result_path
    )