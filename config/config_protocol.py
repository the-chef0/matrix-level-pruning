from typing import Protocol, runtime_checkable
import torch
from torch.nn import Module
from transformers import PreTrainedModel, PreTrainedTokenizer

@runtime_checkable
class ConfigProtocol(Protocol):
    MODEL: PreTrainedModel | Module
    TOKENIZER: PreTrainedTokenizer | None
    DUMMY_INPUT: torch.Tensor
    IMPORTANCES_SAVE_PATH: str | None
    PRUNING_ITERATIONS: int
    PRUNED_MODEL_SAVE_DIR: str | None
    EVALUATE: bool
    EVAL_RESULTS_PATH: str | None
    DEP_GRAPH_ARGS: dict
    BASE_TRANSFORM_TYPES: set[Module]
    TRANSFORM_EXCLUSION_KEYWORDS: set[str]
    BASE_ATTENTION_TYPES: set[Module]
    ATTENTION_CHILD_KEYWORDS: set[str]
    BASE_OPERATION_TYPES: set[Module]
