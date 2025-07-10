import os

import torch
from torch import nn

from infra.chain_collector import collect_groups
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner
from infra.utils.model_utils import ModelUtils

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, 1280)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, n_classes)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x

MODEL = MobileNetV2().to('cuda')
TOKENIZER = None
IMPORTANCES_SAVE_PATH = './importances.csv'
PRUNING_ITERATIONS = 1
PRUNED_MODEL_SAVE_DIR = None
EVALUATE = False
EVAL_RESULTS_PATH = './eval-results.json'
DEP_GRAPH_ARGS = {
    'example_inputs': torch.rand(1,3,512,512).to('cuda'),
    'customized_pruners': {
        nn.ReLU6: OperationPruner(),
    }
}

model_utils = ModelUtils(
    model=MODEL,
    tokenizer=TOKENIZER,
    dep_graph_args=DEP_GRAPH_ARGS
)
print("Model loaded")
print(MODEL)

assert PRUNING_ITERATIONS >= 0
for i in range(PRUNING_ITERATIONS):
    print(f"Iteration {i + 1}")
    # TODO: Make this more efficient
    # Every iteration after the first re-collects groups for the entire model,
    # but we know which modules changed in each pruning iteration.
    # It should suffice to just re-collect groups for the modules affected by pruning.

    print("(Re)building module - name mappings")
    model_utils.build_module_name_mappings()
    print("(Re)building dependency graph")
    model_utils.build_dependency_graph()

    importances_and_groups = collect_groups(
        model_utils,
        iteration=i,
        save_path=IMPORTANCES_SAVE_PATH
    )

    _, group_to_prune = importances_and_groups.pop(0)
    print(f"Pruning group {group_to_prune}")
    group_to_prune.prune()

model_utils.build_dependency_graph()

pruned_model_utils = model_utils

if PRUNED_MODEL_SAVE_DIR:
    pruned_model_utils.tokenizer.save_pretrained(PRUNED_MODEL_SAVE_DIR)
    torch.save(pruned_model_utils.model, os.path.join(PRUNED_MODEL_SAVE_DIR, "model.pth"))
    print(f"Saved pruned model to {PRUNED_MODEL_SAVE_DIR}")
