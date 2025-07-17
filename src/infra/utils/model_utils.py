from torch_pruning.dependency import DependencyGraph

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner
from infra.utils.module_utils.identity_types import IdentityWithGrad

class ModelUtils:
    """A class encapsulating the PyTorch module
    representation of the model, the Torch-Pruning DepGraph representation
    of the model, and auxiliary methods and data structures.
    """

    def __init__(self, cfg: ConfigProtocol):
        self.model = cfg.MODEL.to(cfg.DEVICE)
        self.tokenizer = cfg.TOKENIZER

        self.dep_graph_args = cfg.DEP_GRAPH_ARGS
        self.dep_graph_args['model'] = self.model
        self.dep_graph_args['example_inputs'] = cfg.DUMMY_INPUT.to(cfg.DEVICE)
        self.dep_graph_args['customized_pruners'][IdentityWithGrad] = OperationPruner()
        self.dep_graph = None
        self.build_dependency_graph()

        self.module_to_name = None
        self.name_to_module = None
        self.build_module_name_mappings()
        self.model_modules = None
        self.initialize_module_set()

        self.default_param_count = self.get_param_count()

    def build_module_name_mappings(self):
        print("(Re)building module - name mappings")
        self.module_to_name = {}
        for name, module in self.model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}

    def build_dependency_graph(self):
        print("(Re)building dependency graph")
        self.dep_graph = DependencyGraph().build_dependency(**self.dep_graph_args)

    def initialize_module_set(self):
        self.model_modules = set(self.model.modules())

    def replace_module_by_name(self, module_name, new_module):
        # Split the module name into parts
        parts = module_name.split('.')
        
        # Get the parent module (everything except the last part)
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, parts[-1], new_module)

    def get_param_count(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_sparsity(self):
        return 1 - (self.get_param_count() / self.default_param_count)
