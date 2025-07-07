import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter

from importances_and_groups import collect_groups
from utils.identity_patcher import IdentityPatcher, IdentityWithGrad
from utils.customized_pruners import OperationPruner
from utils.model_utils import ModelUtils

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

        self.conv2b = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2b = nn.BatchNorm2d(64)

        self.conv2c = nn.Conv2d(32, 64, 5, padding=2)
        self.relu2c = nn.ReLU()

        # Block 3 - merge and continue
        self.conv3 = nn.Conv2d(64, 128, 1)
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
        x = xa + xb + xc
        # x = self.add2(self.add1(xa, xb), xc)
        # x = torch.cat([xa, xb], dim=1)

        # Continue
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Output
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

MODEL = ExampleModel().to('cuda')
TOKENIZER = None
DUMMY_INPUT = torch.rand(1,3,256,256).to('cuda')
IMPORTANCES_SAVE_PATH = './importances.csv'
PRUNING_ITERATIONS = 1
PRUNED_MODEL_SAVE_DIR = None
EVALUATE = False
EVAL_RESULTS_PATH = './eval-results.json'
DEP_GRAPH_ARGS = {
    'example_inputs': DUMMY_INPUT,
    'customized_pruners': {
        nn.ReLU: OperationPruner(),
        IdentityWithGrad: OperationPruner()
    }
}

model_utils = ModelUtils(
    model=MODEL,
    dummy_input=DUMMY_INPUT,
    dep_graph_args=DEP_GRAPH_ARGS,
    tokenizer=TOKENIZER
)
print("Model loaded")

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_groups(
    model_utils,
    iteration=0,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[2]
print(g)
g.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_groups(
    model_utils,
    iteration=1,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[1]
print(g)
g.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

importances_and_groups = collect_groups(
    model_utils,
    iteration=2,
    save_path=IMPORTANCES_SAVE_PATH
)

_, g = importances_and_groups[0]
print(g)
g.prune()

model_utils.build_module_name_mappings()
model_utils.build_dependency_graph()

IdentityPatcher(model_utils).patch()
