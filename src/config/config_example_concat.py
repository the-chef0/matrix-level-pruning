from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import modules

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner

torch.manual_seed(0)

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Block 2 - parallel branches
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.relu2a = nn.ReLU()

        self.conv2b = nn.Conv2d(32, 32, 5, padding=2)
        self.bn2b = nn.BatchNorm2d(32)

        self.conv2c = nn.Conv2d(32, 32, 5, padding=2)
        self.relu2c = nn.ReLU()

        # Block 3 - merge and continue
        self.conv3 = nn.Conv2d(128, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Output
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Parallel branches
        xa = self.conv2a(x)
        xa = self.bn2a(xa)
        xa = self.relu2a(xa)

        xb = self.conv2b(x)
        xb = self.bn2b(xb)

        xc = self.conv2c(x)
        xc = self.relu2c(xc)

        # Merge
        x = torch.cat([xa, xb, xc], dim=1)

        # Continue
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Output
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

@dataclass
class Config(ConfigProtocol):
    DEVICE = 'cuda'
    MODEL = ExampleModel()
    TOKENIZER = None
    DUMMY_INPUT = torch.rand(1,3,256,256)
    IMPORTANCES_SAVE_PATH = './importances.csv'
    TARGET_SPARSITY = 0.1
    PRUNED_MODEL_SAVE_DIR = None
    EVALUATE = False
    EVAL_RESULTS_PATH = None
    DEP_GRAPH_ARGS = {
        'customized_pruners': {
            nn.ReLU: OperationPruner()
        }
    }
    TRANSFORM_EXCLUSION_KEYWORDS = set([
        'conv1',
        'avgpool',
        'fc'
    ])
    BASE_ATTENTION_TYPES = set([])
    MHA_PROJECTION_NAME_MAPPING = {}
    BASE_TRANSFORM_TYPES = set([
        modules.Linear,
        modules.conv._ConvNd
    ])
    BASE_OPERATION_TYPES = set([
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
