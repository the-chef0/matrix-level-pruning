import argparse
import gc

from base_model_utils import BaseModelUtils
from eval_pruned import evaluate_pruned
from measure_io_similarity import measure_similarity
from prune_model import prune

import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix-level pruning pipeline')
    
    # IO similarity measurement arguments
    parser.add_argument('--measure_io_similarity', action='store_true', default=False,
                        help='Whether to measure matrix input-output similarity')
    parser.add_argument('--base_model_id', type=str, default=None,
                        help='HuggingFace model ID of the base model')
    parser.add_argument('--similarity_save_path', type=str, default='./similarity.csv',
                        help='Path where to save the measured similarities')

    # Pruning arguments
    parser.add_argument('--prune_model', action='store_true', default=False,
                        help='Whether to prune the model')
    parser.add_argument('--matrices_to_prune', type=str, nargs='+', default='1',
                        help='Either number of top matrices to prune or list of matrix names')
    parser.add_argument('--pruned_model_save_dir', type=str, default=None,
                        help='Directory where to save the pruned model')

    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate the pruned model')
    parser.add_argument('--eval_result_path', type=str, default='./eval-result.json',
                        help='Path where to save the evaluation results')

    args = parser.parse_args()

    # Convert matrices_to_prune to int if possible
    if len(args.matrices_to_prune) == 1:
        try:
            args.matrices_to_prune = int(args.matrices_to_prune[0])
        except ValueError:
            args.matrices_to_prune = args.matrices_to_prune[0].split(',')

    return args

args = parse_args()

base_model_utils = None

if args.base_model_id:
    base_model_utils = BaseModelUtils(args.base_model_id)
    print("Base model loaded")

if args.measure_io_similarity:
    assert base_model_utils is not None
    assert args.similarity_save_path is not None
    measure_similarity(
        base_model_utils,
        save_path=args.similarity_save_path
    )

if args.prune_model:
    assert base_model_utils is not None
    assert args.matrices_to_prune is not None
    assert args.pruned_model_save_dir is not None
    prune(base_model_utils,
          similarity_save_path=args.similarity_save_path,
          matrices_to_prune=args.matrices_to_prune,
          pruned_model_save_dir=args.pruned_model_save_dir
    )

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
