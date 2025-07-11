from typing import Protocol, runtime_checkable
import torch
from torch.nn import Module
from transformers import PreTrainedModel, PreTrainedTokenizer

from enum import Enum, auto

class MHAProjection(Enum):
    Q = auto()
    K = auto()
    V = auto()
    O = auto()

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
    MHA_PROJECTION_NAME_MAPPING: dict[MHAProjection, str]
    BASE_OPERATION_TYPES: set[Module]
