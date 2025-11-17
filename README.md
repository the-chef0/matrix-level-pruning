# Automatic Depth Pruning and Post-Processing for Efficient Deep Learning

There are two sections in this README. One explains how to reproduce the results in the thesis, showing an applied example of this codebase. The other explains the structure of the codebase and how it works overall.

## Reproduction

**First:**
1. Clone this repo
2. Install dependencies using `pip install -r requirements.txt`

**To reproduce the Llama2 results:**
In the corresponding file (see table below), select a point from the `FIGURE_POINTS` dictionary to use in the loop header.

**To reproduce the YOLOv5 results:** 
1. Clone the [YOLOv5 repository](https://github.com/ultralytics/yolov5) and change the `sys.path.append()` line in the YOLO files in the `config` directory to point to the directory of the cloned repo.
2. In the corresponding file, (see table below), select a point from the `FIGURE_POINTS` dictionary to use in the loop header.

**Then:** Run the corresponding file select a point from the `FIGURE_POINTS` dictionary to use in the loop header.

| File                   | Description                                                                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reprod_llama_7b_coarse.py` | Reproduction of our replication of the baseline, Transformer layer pruning method on Llama2 7B.               |
| `reprod_llama_13b_coarse.py` | Reproduction of our replication of the baseline, Transformer layer pruning method on Llama2 13B.               |
| `reprod_llama_7b_fine.py` | Reproduction of our fine-grained depth pruning results on Llama2 7B.                                   |
| `reprod_llama_13b_fine.py`    | Reproduction of our fine-grained depth pruning results on Llama2 13B. |

## Package structure

### Configuration

See `config/config_protocol.py` for a definition of the configuration protocol and see the `config/` directory for concrete examples.

| Field                          | Required                                      | Description                                                                                                                                                                         |
|--------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DEVICE`                       | Yes                                           | The device to load the model onto.                                                                                                                                                  |
| `MODEL`                        | Can be `None` if `TARGET_SPARSITY = 0`                                           | The model to prune. Most PyTorch `nn.Module` objects should be supported, including subclasses like the HuggingFace `AutoModelForCausalLM`.                                         |
| `TOKENIZER`                    | Only if `EVALUATE` is set to `True`, otherwise can be `None`           | The tokenizer for the model.                                                                                                                                                        |
| `DUMMY_INPUT`                  | Yes                                           | Any valid input into the model for tracing the computation graph (see [Why the DepGraph dependency?](#why-the-depgraph-dependency)).                                                                                                                   |
| `IMPORTANCES_SAVE_PATH`        | Can be `None`                                            | A path of the format `/path/to/file.csv` where to save the importance ranking of each identified pruning tree (see [What it does](#what-it-does)).                                                                      |                                                                                     |
| `PRUNED_MODEL_SAVE_DIR`        | Yes                                            | The location to save the pruned model to. If given as `/path/to/model`, the model will be saved under `/path/to/model/model.pth`.                                                   |
| `EVALUATE`                     | Can be `None`                                            | Whether to evaluate the pruned model (TODO: evaluation config).                                                                                                                     |
| `EVAL_RESULTS_PATH`            | Yes                                            | A path of the format `/path/to/file.csv' where to save the results of the evaluation.                                                                                               |
| `DEP_GRAPH_ARGS`               | Probably                                           | Look [here](#additional-notes-on-depgraph-arguments) for more info.                                                                                                                 |
| `TRANSFORM_EXCLUSION_KEYWORDS` | Can be `set([])`                                            | (Sub)strings of variable names of transforms that should not be considered for pruning (e.g. embedding layers, classification heads...).                                            |
| `BASE_ATTENTION_TYPES`         | Only in attention-based models; otherwise can be `set([])`                | Defines which module type(s) should be considered parent attention module(s) (see [the Pruning Tree data structure](#the-pruning-tree-data-structure))                                                                                                     |
| `MHA_PROJECTION_NAME_MAPPING`  | Only in attention-based models; otherwise can be `set([])`               | A mapping determining which variable names the code should look for within the attention module(s) to identify the Q, K, V and O projections.                                       |
| `BASE_TRANSFORM_TYPES`         | Yes, defaults in given config file likely reusable | Defines which module types should be considered transforms (see [the Pruning Tree data structure](#the-pruning-tree-data-structure)).                                               |
| `BASE_OPERATION_TYPES`         | Yes, but defaults in given config file likely reusable                                           | Defines which modules types should be considered activation functions, normalization functions, etc. (see  [the Pruning Tree data structure](#the-pruning-tree-data-structure)) |

#### Additional notes on DepGraph arguments

You will most likely need to specify the following arguments:
 - `output_transform`: HuggingFace models often do not directly output the prediction logits, but a wrapper that also contains past KV data for caching and so on. The `output_transform` must be a callable that extracts the logits from whatever the model outputs.
 - `customized_pruners`: This is used to specify which module types should be considered nodes in the DepGraph data structure. They get detected via these pruner classes. By default, it ignores activation and normalization modules. See the Torch-Pruning repo to see what it does detect by default. For unparametrized operations like ReLU, you can use the placeholder `OperationPruner` from `infra/utils/dep_graph_utils/custom_pruners.py`. The file also contains other pruners that I used during development, maybe some will be useful to you. If not, you might need to define your own.

For more info, please see the [Torch-Pruning repo](https://github.com/VainF/Torch-Pruning/). To understand why this is required, see [Why the DepGraph Dependency?](#why-the-depgraph-dependency).

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

### The pruning tree data structure

A pruning tree consists of a *parameter subtree* and an *operation subtree*. The root of each tree is either a *transform layer* - meaning linear layer or convolution layer (see [`BASE_TRANSFORM_TYPES`](#configuration)) - or an attention head (see [`BASE_ATTENTION_TYPES`](#configuration)). The root is the primary object to prune to avoid the latency associated with computing it, and the rest of the tree consists of objects that need to removed along with it.

#### Transform layers

For a tree rooted at a transform layer, the parameter subtree contains the root and includes rows/columns of parameter matrices that need to be adjusted to make sure we don't break the dimensions of the model once the root is removed.

For example, if a linear layer $l$ has input dimension 2048 and output dimension 2048, we can remove it without dimension conflicts because the inputs into the root produce a representation of size 2048, and the layers that come after the root also expect an input of size 2048.

However, if the linear layer has e.g. input dimension 2048 and output dimension 4096, this no longer works. The layer $l+1$ that comes after the root expects an input of size 4096, but by removing the root, the input passes directly through and retains its size of 2048. To fix this, we *width-prune* this dependency, i.e. remove columns from the parameter matrix (corresponding to input dimensions) of $l+1$. By width-pruning 2048 columns, $l+1$ becomes able to accept an input of size 2048. **Overall, the parameter subtree consists of the root $l$, and as a dimensional dependency, the 2048 least important columns in the parameter matrix of $l+1$.** The case of a linear layer that reduces dimensions rather than expands is analogous.

Lastly, the operation subtree consists of any activation functions, normalization functions and other such operations that are coupled to the root. "Coupled to the root" means they receive an input/output only from the root. 

Everything collected in the tree is considered for removal. The parameter subtree contains the root, which is the primary unit of computation we want to get rid of (and also save on memory by removing its parameters), along with dependent rows/columns in adjacent parameter matrices. The operation subtree consists of functions that we no longer need once the root is removed, so we can remove them too for a small, additional latency boost.

#### Attention heads

**Note:** This codebase supports pruning attention heads, but in the final thesis, this feature was not used. Instead, we pruned full attention layers by pruning all the heads within them. This is done in the reproduction files too.

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

### Why the DepGraph dependency?
Much of the code is built on top of the [Torch-Pruning repo](https://github.com/VainF/Torch-Pruning/), which implements the DepGraph data structure. Torch-Pruning is a structured, width pruning implementation, and the DepGraph data structure is used to answer the following question: "If I adjust the input/output dimension of a layer by width-pruning its parameter matrix, what dimension-dependent parameters in other layers do I also have to prune?"

In my implementation, pruning the root of a transform pruning tree is implemented as replacing it with an identity function, which is equivalent on a dimensional level to changing the layer's output dimension to be equal to its input dimension. The DepGraph helps identify the affected, dimension-dependent layers.

Additionally, Torch-Pruning creates this DepGraph data structure by traversing the PyTorch Autograd graph. This provided a convenient computation graph representation of the model that I used to identify operations that belong in pruning trees, identify redundant arithmetic operations and so on.