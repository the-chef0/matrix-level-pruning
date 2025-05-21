# matrix-level-pruning

## Example commands
```
python3 entrypoint.py \
--base_model_id meta-llama/Llama-3.2-1B \
--measure_io_similarity \
--similarity_save_path ./similarity.csv \
--prune_model \
--matrices_to_prune 1 \
--pruned_model_save_dir /home/michal/hf-models/llama3-top1 \
--evaluate \
--eval_result_path ./result.json
```

```
python3 entrypoint.py \
--base_model_id meta-llama/Llama-3.2-1B \
--measure_io_similarity \
--similarity_save_path ./similarity.csv \
--prune_model \
--matrices_to_prune model.layers.4.mlp.up_proj,model.layers.10.mlp.gate_proj \
--pruned_model_save_dir /home/michal/hf-models/llama3-2matrices \
--evaluate \
--eval_result_path ./result.json
```

## Explanations of arguments
| Argument                | Explanation                                                                                                                                                                    |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `base_model_id`         | HuggingFace model ID of the base model for measuring matrix similarities and/or pruning.                                                                                       |
| `measure_io_similarity` | Whether to measure and generate a file containing input/output similarities of each matrix. Currently this is only done with the calibration dataset `arcee-ai/sec-data-mini`. |
| `similarity_save_path`  | The path where the similarity measurements will be saved, formatted as `/path/to/filename.csv`. The path will be created if it does not exist.                                 |
| `prune_model`           | Whether to prune the base model.                                                                                                                                               |
| `matrices_to_prune`     | Either an integer $k$ denoting that the matrices with the $k$ highest input/output similarities should be pruned, or a comma-separated list of module names.                   |
| `pruned_model_save_dir` | Path to a directory to store the pruned model in. The path will be created if it does not exist.                                                                               |
| `evaluate`              | Whether to evaluate the performance of the pruned model. Currently this is only done on the HellaSwag benchmark.                                                               |
| `eval_results_path`     | The path where the evaluation results will be saved, formatted as `/path/to/filename.json`. The path will be created if it does not exist.                                     |