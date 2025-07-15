# Computation Graph Pruning: A General Method Applied to Language Models

## Usage guide

1. Clone this repo
2. Configure settings in `config/config.py` (see [below](#configuration) for more info)
3. Run `python entrypoint.py`

To understand how the code is structured, see [Package structure](#package-structure). To understand how that translates to a higher, conceptual level, see [What it does](#what-it-does).

### Examples

To see an example, run `python [example_filename].py`.

| File                   | Description                                                                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `example_llama_mlp.py` | Demonstrates 2 pruning iterations in Llama3 1B that also create a redundant multiplication situation to demonstrate artithmetic identity patching.               |
| `example_mobilenet.py` | Demonstrates 5 pruning iterations in MobileNetV2 to demonstrate applicability to general PyTorch models, not just HuggingFace LMs.                                  |
| `example_concat.py`    | Demonstrates 3 pruning iterations in an invented example model that also create a redundant concatenation situation to demonstrate concatenative identity patching. |

### Configuration

See `config/config_protocol.py` for a definition of the configuration protocol and `config/config.py` for a concrete example.

| Field                          | Required                                      | Description                                                                                                                                                                         |
|--------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DEVICE`                       | Yes                                           | The device to load the model onto.                                                                                                                                                  |
| `MODEL`                        | Yes                                           | The model to prune. Most PyTorch `nn.Module` objects should be supported, including subclasses like the HuggingFace `AutoModelForCausalLM`.                                         |
| `TOKENIZER`                    | Only if `EVALUATE` is set to `True`, otherwise can be `None`           | The tokenizer for the model.                                                                                                                                                        |
| `DUMMY_INPUT`                  | Yes                                           | Any valid input into the model for tracing the computation graph (see [Why the DepGraph dependency?](#why-the-depgraph-dependency)).                                                                                                                   |
| `IMPORTANCES_SAVE_PATH`        | Can be `None`                                            | A path of the format `/path/to/file.csv` where to save the importance ranking of each identified pruning tree (see [What it does](#what-it-does)).                                                                      |
| `PRUNING_ITERATIONS`           | Yes                                           | The number of times to repeat the pruning tree collection pass (see [Putting it all together](#putting-it-all-together)).                                                                                        |
| `PRUNED_MODEL_SAVE_DIR`        | Can be `None`                                            | The location to save the pruned model to. If given as `/path/to/model`, the model will be saved under `/path/to/model/model.pth`.                                                   |
| `EVALUATE`                     | Can be `None`                                            | Whether to evaluate the pruned model (TODO: evaluation config).                                                                                                                     |
| `EVAL_RESULTS_PATH`            | Can be `None`                                            | A path of the format `/path/to/file.csv' where to save the results of the evaluation.                                                                                               |
| `DEP_GRAPH_ARGS`               | Probably                                           | Look [here](#additional-notes-on-depgraph-arguments) for more info.                                                                                                                 |
| `BASE_TRANSFORM_TYPES`         | Defaults in given config file likely reusable | Defines which module types should be considered transforms (see [the Pruning Tree data structure](#the-pruning-tree-data-structure)).                                               |
| `TRANSFORM_EXCLUSION_KEYWORDS` | Can be `set([])`                                            | (Sub)strings of variable names of transforms that should not be considered for pruning (e.g. embedding layers, classification heads...).                                            |
| `BASE_ATTENTION_TYPES`         | Only in attention-based models; otherwise can be `set([])`.                | Defines which module type(s) should be considered parent attention module(s) (see [the Pruning Tree data structure](#the-pruning-tree-data-structure))                                                                                                     |
| `MHA_PROJECTION_NAME_MAPPING`  | Only in attention-based models; otherwise can be `set([])`.                | A mapping determining which variable names the code should look for within the attention module(s) to identify the Q, K, V and O projections.                                       |
| `BASE_OPERATION_TYPES`         | Yes                                           | Defines which modules types should be considered activation functions, normalization functions, etc. (see  [the Pruning Tree data structure](#the-pruning-tree-data-structure)) |

#### Additional notes on DepGraph arguments

You will most likely need to specify the following arguments:
 - `output_transform`: HuggingFace models often do not directly output the prediction logits, but a wrapper that also contains past KV data for caching and so on. The `output_transform` must be a callable that extracts the logits from whatever the model outputs.
 - `customized_pruners`: This is used to specify which module types should be considered nodes in the DepGraph data structure. They get detected via these pruner classes. By default, it ignores activation and normalization modules. See the Torch-Pruning repo to see what it does detect by default. For unparametrized operations like ReLU, you can use the placeholder `OperationPruner` from `infra/utils/dep_graph_utils/custom_pruners.py`. The file also contains other pruners that I used during development, maybe some will be useful to you. If not, you might need to define your own.

For more info, please see the [Torch-Pruning repo](https://github.com/VainF/Torch-Pruning/). To understand why this is required, see [Why the DepGraph Dependency?](#why-the-depgraph-dependency).

## Package structure

The code uses two simultaneous representations of the model and switches between them for different purposes. The table below describes how the code is organized to reflect this.

| Package                       | Description                                                                                                                                                                               |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `root`                        | Contains the entrypoint file and example files.                                                                                                                                           |
| `config`                      | Contains the configuration protocol definition, the main config file, and the config files pertaining to the examples.                                                                    |
| `infra`                       | Contains all the logic.                                                                                                                                                                   |
| `infra/passes`                | Defines the passes (see [What it does](#what-it-does)).                                                                                                                                   |
| `infra/pruning_tree_types`    | Defines the data structures used for pruning (see [The pruning tree data structure](#the-pruning-tree-data-structure)).                                                                   |
| `infra/utils/dep_graph_utils` | Contains helper functions and classes that operate primarily on the DepGraph representation of the model (see [Why the DepGraph dependency?](#why-the-depgraph-dependency)).              |
| `infra/utils/module_utils`    | Contains helper functions and classes that operate primarily on the PyTorch Module representation of the model.                                                                           |
| `infra/utils/model_utils.py`  | Defines a class that encapsulates the DepGraph representation of the model, the PyTorch Module representation of the model, and manages interoperability between the two representations. |

## What it does

(This section describes the concepts with minimal mathematical formality. The formal version will probably end up in the thesis report rather than here.)

At the time of writing this, state-of-the-art LLM pruning works by removing (depth-pruning) transformer/attention/MLP layers that are deemed unimportant by some metric. This leads to a reduction in memory footprint, but importantly, it also leads to reduced inference latency which is crucial for deployment at scale. Compared to structured and unstructured pruning which saves on latency by reducing memory accesses, this SOTA depth pruning of entire transformer/attention/MLP layers saves on latency by skipping these computations entirely. However, taking these large composite blocks to be the pruning units yields a relatively small search space. My work expands this search space by zooming in on the finest level on which this idea of "skipping computations entirely" applies.

The main functionality consists of several iterations of the *pruning tree collection* pass and one *identity patching pass*

### The pruning tree data structure

A pruning tree consists of a *parameter subtree* and an *operation subtree*. The root of each tree is either a *transform layer* - meaning linear layer or convolution layer (see [`BASE_TRANSFORM_TYPES`](#configuration)) - or an attention head (see [`BASE_ATTENTION_TYPES`](#configuration)). The root is the primary object to prune to avoid the latency associated with computing it, and the rest of the tree consists of objects that need to removed along with it.

#### Transform layers

For a tree rooted at a transform layer, the parameter subtree contains the root and includes rows/columns of parameter matrices that need to be adjusted to make sure we don't break the dimensions of the model once the root is removed.

For example, if a linear layer $l$ has input dimension 2048 and output dimension 2048, we can remove it without dimension conflicts because the inputs into the root produce a representation of size 2048, and the layers that come after the root also expect an input of size 2048.

However, if the linear layer has e.g. input dimension 2048 and output dimension 4096, this no longer works. The layer $l+1$ that comes after the root expects an input of size 4096, but by removing the root, the input passes directly through and retains its size of 2048. To fix this, we *width-prune* this dependency, i.e. remove columns from the parameter matrix (corresponding to input dimensions) of $l+1$. By width-pruning 2048 columns, $l+1$ becomes able to accept an input of size 2048. **Overall, the parameter subtree consists of the root $l$, and as a dimensional dependency, the 2048 least important columns in the parameter matrix of $l+1$.** The case of a linear layer that reduces dimensions rather than expands is analogous.

Lastly, the operation subtree consists of any activation functions, normalization functions and other such operations that are coupled to the root. "Coupled to the root" means they only receive an input from the root. 

Everything collected in the tree is considered for removal. The parameter subtree contains the root, which is the primary unit of computation we want to get rid of (and also save on memory by removing its parameters), along with dependent rows/columns in adjacent parameter matrices. The operation subtree consists of functions that we no longer need once the root is removed, so we can remove them too for a small, additional latency boost.

#### Attention heads

For a tree rooted at an attention head $i$, the parameter subtree consists of rows in the Q, K and V projection matrices, and columns in the O projection matrix, that correspond to attention head $i$. There is no need for dimensional dependencies in the subtree, because contrary to what I was able to find in the literature, it is possible to prune attention heads without breaking dimensions in other parts of the model.

Briefly, multihead attention can be formalized as:
```math
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \mathbf{W}^O
```
We reduce the number of heads by width-pruning rows of the Q, K and V projection matrices. This reduces the dimension of the result of the concat operation, so then, we width-prune the columns of the O projection matrix to adapt it to the new input dimension. The rest of the model remains unaffected because the output dimension of the O projection matrix stays the same.

The operation subtree for trees rooted at attention heads exists only if the root is the last remaining head in the attention layer (all others have been pruned). Otherwise, the other attention heads still depend on the operations being there.

With attention heads, the idea of "skipping computations" does not manifest itself until we can remove the final attention head in an attention layer, thereby removing the entire layer. However, to the best of my knowledge, only my work can prune heads in a precise way that doesn't require dimension adjustments in other parts of the model. Additionally, by pruning heads, we can still improve latency by reducing memory access and shrinking the KV cache.

### Identity patching

Suppose your model contains some element-wise arithmetic operations, e.g. a gated MLP block. Let us formalize this as:
```math
\textsf{SiLU}(\mathbf{X} \mathbf{W}^G) * \mathbf{X} \mathbf{W}^U
```
Now suppose we prune out a tree containing $\mathbf{W}^G$ and $\textsf{SiLU}$, and a tree containing $\mathbf{W}^U$. This results in
```math
\textsf{id}(\textsf{id}(\mathbf{X})) * \textsf{id}(\mathbf{X}) = \mathbf{X} * \mathbf{X}
```
where $\textsf{id}$ is an identity operation that just copies the input to the output. By pruning the above layers and operations, we've created an unnecessary multiplication-by-self. The identity patching pass gets rid of these situations to save on a bit more latency.

Since the element-wise arithmetic operation in question is a multiplication, the identity patcher replaces one of the identity function operands with a function that simply returns the multiplicative identity element - an all-ones tensor with a compatible shape:
```math
\textsf{id}(\textsf{id}(\mathbf{X})) * \mathbf{1} = \mathbf{X}
```
When compiling the model for inference, the compiler will likely detect these redundant identity operations and the redundant multiplication and optimize them away.

### Putting it all together

With both passes described in detail, we can look at an overview of the top-level functionality:

1. Do a pruning tree collection pass - obtain one tree per transform layer and attention head as a root.
2. Measure the total importance of each tree.
3. Prune out the least important tree.
4. If the target parameter count reduction has not yet been reached, go to 1. Otherwise go to 5.
5. Do an identity patching pass.

The model will then be ready for fine-tuning, compilation for inference, and other downstream tasks.

### Why the DepGraph dependency?
Much of the code is built on top of the [Torch-Pruning repo](https://github.com/VainF/Torch-Pruning/), which implements the DepGraph data structure. Torch-Pruning is a structured, width pruning implementation, and the DepGraph data structure is used to answer the following question: "If I adjust the input/output dimension of a layer by width-pruning its parameter matrix, what dimension-dependent parameters in other layers do I also have to prune?"

In my implementation, pruning the root of a transform pruning tree is implemented as replacing it with an identity function, which is equivalent on a dimensional level to changing the layer's output dimension to be equal to its input dimension. The DepGraph helps identify the affected, dimension-dependent layers.

Additionally, Torch-Pruning creates this DepGraph data structure by traversing the PyTorch Autograd graph. This provided a convenient computation graph representation of the model that I used to identify operations that belong in pruning trees, identify redundant arithmetic operations and so on.

# TODO
 - [x] Fix attention head grouping algo
 - [x] Make naming in attention_pruning_tree consistent with transform_pruning_tree
 - [x] Clean up identity_patcher, see if any dep_graph_search_utils functions can be repurposed
 - [x] Use config object in eval
 - [ ] Improve defaults in provided config file
 - [x] Write a usage guide
 - [ ] Finish docstrings
 - [ ] Make sure type hints are everywhere
 - [x] Update examples
 - [x] Make sure everything in dep_graph_search_utils takes args on the dep graph level and returns types on the dep graph level
 - [x] Parametrize device in config
 - [x] Make a pruning listener that rebuilds data structures in model_utils every time something is pruned
 - [ ] (Low priority) There should be a way to make general DFS functions like `find_first_node_by_cond`, `collect_nodes_by_cond` and `count_nodes_by_cond` and pass the conditions as lambdas, but the work needed to make this kind of abstraction might not be worth it at all.
 - [ ] (Low priority) Maybe there is some stuff that repeats in both transform_pruning_tree and attention_pruning_tree that I could take care of in the pruning_tree superclass. Look into class methods, static methods, etc. Could also do a template method prune in the superclass that calls do_prune (concrete implementation) followed by call_post_prune_listeners.
