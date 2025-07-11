# matrix-level-pruning

TODO:
 - [x] Fix attention head grouping algo
 - [x] Make naming in attention_pruning_tree consistent with transform_pruning_tree
 - [x] Clean up identity_patcher, see if any dep_graph_search_utils functions can be repurposed
 - [x] Use config object in eval
 - [ ] Write a usage guide
 - [ ] Finish docstrings
 - [ ] Make sure type hints are everywhere
 - [ ] Update examples
 - [x] Make sure everything in dep_graph_search_utils takes args on the dep graph level and returns types on the dep graph level
 - [x] Parametrize device in config
 - [x] Make a pruning listener that rebuilds data structures in model_utils every time something is pruned
 - [ ] (Low priority) There should be a way to make general DFS functions like `find_first_node_by_cond`, `collect_nodes_by_cond` and `count_nodes_by_cond` and pass the conditions as lambdas, but the work needed to make this kind of abstraction might not be worth it at all.
 - [ ] (Low priority) Maybe there is some stuff that repeats in both transform_pruning_tree and attention_pruning_tree that I could take care of in the pruning_tree superclass. Look into class methods, static methods, etc.
