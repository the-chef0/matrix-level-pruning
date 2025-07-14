# Computation Graph Pruning: A General Method Applied to Language Models

## Usage guide

1. Clone this repo
2. Configure settings in `config/config.py` (see [below](#configuration-fields) for more info)
3. Run `python entrypoint.py`

### Configuration fields (WIP)

See `config/config_protocol.py` for a definition of the configuration protocol and `config/config.py` for a concrete example.

| Field                   | Required                            | Description                                                                                                                                 |
|-------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `DEVICE`                | Yes                                 | The device to load the model onto.                                                                                                          |
| `MODEL`                 | Yes                                 | The model to prune. Most PyTorch `nn.Module` objects should be supported, including subclasses like the HuggingFace `AutoModelForCausalLM`. |
| `TOKENIZER`             | Only if `EVALUATE` is set to `True` | The tokenizer for the model.                                                                                                                |
| `DUMMY_INPUT`           | Yes                                 | Any valid input into the model for tracing the computation graph.                                                                           |
| `IMPORTANCES_SAVE_PATH` | No                                  | A path of the format `/path/to/file.csv` where to save the importance ranking of each identified pruning tree.                              |
| `PRUNING_ITERATIONS`    | Yes                                 | The number of times to repeat the pruning tree collection pass (see [below](#how-it-works)).                                                |
| `PRUNED_MODEL_SAVE_DIR` | No                                  | The location to save the pruned model to. If given as `/path/to/model`, the model will be saved under `/path/to/model/model.pth`.           |
| `EVALUATE`              | No                                  | Whether to evaluate the pruned model (TODO: evaluation config).                                                                             |
| `EVAL_RESULTS_PATH`     | No                                  | A path of the format `/path/to/file.csv' where to save the results of the evaluation.                                                       |

#### Additional notes on dependency graph arguments (TODO)

## How it works (WIP)

### The core idea

The goal of this work is to optimize the latency of a a model defined in PyTorch by pruning redundant nodes in its computation graph $\mathcal{G}$. On an implementation level, this is done by analyzing the model through its DepGraph representation and modifying it on the level of its PyTorch module structure, such that once the model is compiled for inference (using ONNX or Torch Export), the redundant computations do not take place.

### Notational preliminaries

Borrowing notation from DepGraph, we decompose a network $\mathcal{F}(x;w)$ into a set of components $\mathcal{F} = \{ f_1, f_2, ..., f_L \}$ where each component refers to either a parametrized layer such as a linear layer or a convolution, or a non-parametrized operation such as an activation function or a residual addition. Then, we denote the input and output of component $f_i$ as $f_i^{-}$ and $f_i^{+}$ respectively.

In a deviation from the notation in DepGraph, we represent the network $\mathcal{F}$ as a directed computation graph $\mathcal{G}$, where
 - **Nodes** are tuples $(f_i^{-}, f_i^{+})$.
 - **Edges** order the sequence of dependent operations. An edge $(f_i^{-}, f_i^{+}) \rightarrow (f_j^{-}, f_j^{+})$ exists if computing $f_j$ requires as an operand the immediate result of computing $f_i$.

For example, an MLP structure can be represented as 

```math
(\textsf{Linear}_i^{-}, \textsf{Linear}_i^{+}) \rightarrow (\textsf{ReLU}_i^{-}, \textsf{ReLU}_i^{+}) \rightarrow (\textsf{Linear}_{i+1}^{-}, \textsf{Linear}_{i+1}^{+}) \rightarrow (\textsf{ReLU}_{i+1}^{-}, \textsf{ReLU}_{i+1}^{+}) \rightarrow (\textsf{Linear}_{i+2}^{-}, \textsf{Linear}_{i+2}^{+})
```

Now, DepGraph describes the notion of *inter-layer dependencies* $f_i^+ \Leftrightarrow f_j^-$. With our computation graph formulation, this arises in connected layers where $(f_i^{-}, f_i^{+}) \rightarrow (f_j^{-}, f_j^{+})$. 

In the above example, there exist inter-layer dependencies $\textsf{Linear}_i^{+} \Leftrightarrow \textsf{ReLU}_{i}^{-}$ and $\textsf{ReLU}_{i}^{+} \Leftrightarrow \textsf{Linear}_{i+1}^{-}$. To formalize why this is the case, let us define a dimension function $\textsf{dim} : \mathcal{F} \longrightarrow \mathbb{N}^+$ that maps each component in $\mathcal{F}$ to the size of its representation dimension. In the case of linear layers, this refers to the size of the hidden dimension. In the context of convolution layers, this refers to the size of the channel dimension. In general, the following holds:

```math
(f_i^{-}, f_i^{+}) \rightarrow (f_j^{-}, f_j^{+}) \implies \textsf{dim}(f_i^+) = \textsf{dim}(f_j^-)
```

In other words, if the result of computing $f_i$ is directly used an operand for computing $f_j$, then the output size of $f_i$ must match the input size of $f_j$.

### Algorithm overview

### The Pruning Tree data structure

A pruning tree consists of
 - a parameter subtree
 - and an operation subtree.

In most cases, the parameter subtree will have a root. 

A root has to be a *transform node* $(t^-, t^+)$ in $\mathcal{G}$, which we take to mean a **linear layer** or a **convolution layer**. The goal is to prune $(t^-, t^+)$ by replacing it with an identity operation to turn the node into $(\textsf{id}_t^-, \textsf{id}_t^+)$.

Since the identity operation is dimension-preserving, we have $\textsf{dim}(\textsf{id}^-) = \textsf{dim}(\textsf{id}^+)$. This is not a problem if $\textsf{dim}(t^-) = \textsf{dim}(t^+)$, i.e. the layer contains a square parameter matrix that maps between spaces of equal dimensions. However, if $\textsf{dim}(t^-) \neq \textsf{dim}(t^+)$, we must consider the following:

### Usage of the VainF/Torch-Pruning repo

# TODO
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
 - [ ] (Low priority) Maybe there is some stuff that repeats in both transform_pruning_tree and attention_pruning_tree that I could take care of in the pruning_tree superclass. Look into class methods, static methods, etc. Could also do a template method prune in the superclass that calls do_prune (concrete implementation) followed by call_post_prune_listeners.
