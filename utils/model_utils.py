from torch import nn
from torch_pruning.dependency import DependencyGraph

class ModelUtils:
    def __init__(self, model: nn.Module = None, tokenizer = None, dep_graph_args: dict = None):
        self.model = model
        self.tokenizer = tokenizer
        self.dep_graph_args = dep_graph_args
        self.dep_graph_args['model'] = self.model
        self.dep_graph = None
        self.module_to_name = None
        self.name_to_module = None

    def build_module_name_mappings(self):
        self.module_to_name = {}
        for name, module in self.model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}

    def build_dependency_graph(self):
        self.dep_graph = DependencyGraph().build_dependency(**self.dep_graph_args)
    