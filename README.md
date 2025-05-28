# matrix-level-pruning

## Example commands
```
python3 entrypoint.py \
--base_model_id meta-llama/Llama-3.2-1B \
--importances_save_path ./importances.csv \
--pruning_iterations 2 \
--pruned_model_save_dir /home/michal/hf-models/llama3-top1 \
--evaluate \
--eval_result_path ./result.json
```

## Explanations of arguments
| Argument                | Explanation                                                                                                                                                                    |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `base_model_id`         | HuggingFace model ID of the base model for measuring matrix similarities and/or pruning.                                                                                       |
| `similarity_save_path`  | The path where the importance measurements will be saved, formatted as `/path/to/filename.csv`. The path will be created if it does not exist.                                 |                                                                                                                                         |
| `pruning_iterations`     | An integer denoting how many times to perform the following process: Collect pruning groups, rank their importances, and prune the least important one. If this is set to 0, no pruning will happen - only evaluation (if applicable).                  |
| `pruned_model_save_dir` | Path to a directory to store the pruned model in. The path will be created if it does not exist.                                                                               |
| `evaluate`              | Whether to evaluate the performance of the pruned model. Currently this is only done on the HellaSwag benchmark.                                                               |
| `eval_results_path`     | The path where the evaluation results will be saved, formatted as `/path/to/filename.json`. The path will be created if it does not exist.                                     |