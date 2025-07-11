# matrix-level-pruning

TODO:
 - [x] Fix attention head grouping algo
 - [x] Make naming in attention_pruning_tree consistent with transform_pruning_tree
 - [ ] Clean up identity_patcher, see if any dep_graph_search_utils functions can be repurposed
 - [ ] Use config object in eval
 - [ ] Write a usage guide
 - [ ] Finish docstrings
 - [ ] Make sure type hints are everywhere
 - [ ] Update examples
 - [ ] Make sure everything in dep_graph_search_utils takes args on the dep graph level and returns types on the dep graph level
 - [ ] Parametrize device in config
 - [ ] Make a pruning listener that rebuilds data structures in model_utils every time something is pruned
