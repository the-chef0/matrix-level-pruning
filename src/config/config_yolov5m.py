from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import modules
import sys
sys.path.append("/path/to/yolov5")
from models.common import DetectMultiBackend

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner

@dataclass
class Config(ConfigProtocol):
    DEVICE = 'cuda'
    MODEL = DetectMultiBackend('yolov5m.pt', fp16=False)
    TOKENIZER = None
    DUMMY_INPUT = torch.randn(1, 3, 224, 224).to(DEVICE)
    IMPORTANCES_SAVE_PATH = None
    IMPORTANCES_SAVE_PATH = None
    TARGET_SPARSITY = 0.1
    PRUNED_MODEL_SAVE_DIR = './pruned'
    EVALUATE = True
    EVAL_RESULTS_PATH = None
    DEP_GRAPH_ARGS = {
        'output_transform': lambda output: output[0],
        'customized_pruners': {
            nn.SiLU: OperationPruner(),
        }
    }
    BASE_TRANSFORM_TYPES = set([
        modules.Linear,
        modules.conv._ConvNd
    ])
    TRANSFORM_EXCLUSION_KEYWORDS = set([
        '24.m'
    ])
    BASE_ATTENTION_TYPES = set([
    ])
    MHA_PROJECTION_NAME_MAPPING = {
    }
    BASE_ACT_TYPES = set([
        nn.ELU,
        nn.Hardshrink,
        nn.Hardsigmoid,
        nn.Hardtanh,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.PReLU,
        nn.ReLU,
        nn.ReLU6,
        nn.RReLU,
        nn.SELU,
        nn.CELU,
        nn.GELU,
        nn.Sigmoid,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Softshrink,
        nn.Softsign,
        nn.Tanh,
        nn.Tanhshrink,
        nn.Threshold,
        nn.GLU,
        nn.Softmin,
        nn.Softmax,
        nn.Softmax2d,
        nn.LogSoftmax,
        nn.AdaptiveLogSoftmaxWithLoss,
    ])
    BASE_NORM_TYPES = set([
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LazyInstanceNorm1d,
        nn.LazyInstanceNorm2d,
        nn.LazyInstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
        nn.RMSNorm,
    ])