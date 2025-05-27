# matrix-level-pruning

## Example commands
```
python3 entrypoint.py \
--base_model_id meta-llama/Llama-3.2-1B \
--measure_importances \
--importances_save_path ./similarity.csv \
--prune_model \
--groups_to_prune 1 \
--pruned_model_save_dir /home/michal/hf-models/llama3-top1 \
--evaluate \
--eval_result_path ./result.json
```

## Explanations of arguments
| Argument                | Explanation                                                                                                                                                                    |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `base_model_id`         | HuggingFace model ID of the base model for measuring matrix similarities and/or pruning.                                                                                       |
| `measure_importances` | Whether to form groups and measure the importance of each group. |
| `similarity_save_path`  | The path where the importance measurements will be saved, formatted as `/path/to/filename.csv`. The path will be created if it does not exist.                                 |
| `prune_model`           | Whether to prune the base model.                                                                                                                                               |
| `groups_to_prune`     | An integer $k$ denoting that the groups with the $k$ lowest importances should be pruned.                   |
| `pruned_model_save_dir` | Path to a directory to store the pruned model in. The path will be created if it does not exist.                                                                               |
| `evaluate`              | Whether to evaluate the performance of the pruned model. Currently this is only done on the HellaSwag benchmark.                                                               |
| `eval_results_path`     | The path where the evaluation results will be saved, formatted as `/path/to/filename.json`. The path will be created if it does not exist.                                     |