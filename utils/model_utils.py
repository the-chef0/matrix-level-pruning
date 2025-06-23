from torch import nn, Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch_pruning.dependency import DependencyGraph

class ModelUtils:
    def __init__(self, model: nn.Module, dummy_input: Tensor, dep_graph_args: dict, tokenizer = None):
        self.model = model
        self.tokenizer = tokenizer
        self.dep_graph_args = dep_graph_args
        self.dep_graph_args['model'] = self.model
        self.dep_graph = None
        self.dummy_input = dummy_input
        self.module_to_name = None
        self.name_to_module = None
        self.ir_graph = None

    def build_module_name_mappings(self):
        self.module_to_name = {}
        for name, module in self.model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}

    def build_dependency_graph(self):
        self.dep_graph = DependencyGraph().build_dependency(**self.dep_graph_args)

    def build_ir_graph(self):
        self.model = make_fx(self.model)(self.dummy_input)
        self.ir_graph = self.model.graph
