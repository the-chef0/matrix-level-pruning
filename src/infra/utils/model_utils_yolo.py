"""
Quick fiX with this specialized file, ModelUtils should have been an interface or abstract class
"""

from torch.nn import Module
from torch.nn import Module
from torch_pruning.dependency import DependencyGraph

from yolov5.utils.loss import ComputeLoss
from yolov5.utils.general import check_dataset, check_img_size
from yolov5.utils.dataloaders import create_dataloader

from config.config_protocol import ConfigProtocol
from infra.utils.dep_graph_utils.custom_pruners import OperationPruner
from infra.utils.module_utils.identity_types import IdentityWithGrad

class ModelUtils:
    """A class encapsulating the PyTorch module
    representation of the model, the Torch-Pruning DepGraph representation
    of the model, and auxiliary methods and data structures.
    """

    def __init__(self, cfg: ConfigProtocol):
        self.cfg = cfg
        self.model = cfg.MODEL.to(cfg.DEVICE)
        if cfg.TOKENIZER:
            self.tokenizer = cfg.TOKENIZER
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
            self.tokenizer.padding_size = 'left'

        for p in self.model.parameters():
            p.requires_grad_(True)

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

        self.estimate_gradients()

    def build_module_name_mappings(self) -> None:
        print("(Re)building module - name mappings")
        self.module_to_name = {}
        for name, module in self.model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}

    def build_dependency_graph(self) -> None:
        print("(Re)building dependency graph")
        self.dep_graph = DependencyGraph().build_dependency(**self.dep_graph_args)

    def initialize_module_set(self) -> None:
        # Creates a set of modules for fast lookup.
        self.model_modules = set(self.model.modules())

    def replace_module_by_name(self, module_name: str, new_module: Module) -> None:
        # Split the module name into parts
        parts = module_name.split('.')
        
        # Get the parent module (everything except the last part)
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, parts[-1], new_module)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_sparsity(self) -> float:
        return 1 - (self.get_param_count() / self.default_param_count)
    
    def estimate_gradients(self) -> None:
        model = self.model.model

        # Dataset configuration (uses the built-in coco.yaml)
        data = check_dataset("/path/to/dataset.yaml")
        imgsz = check_img_size(640, s=model.stride.max())

        # Create dataloader (no augmentation for val)
        dataloader = create_dataloader(
            path=data["val"],  # or data['train']
            imgsz=imgsz,
            batch_size=2,
            stride=int(model.stride.max()),
            pad=0.5,
            rect=False,
            workers=4
        )[0]

        compute_loss = ComputeLoss(model)

        for imgs, targets, paths, shapes in dataloader:
            imgs = imgs.to(self.cfg.DEVICE).float() / 255.0
            targets = targets.to(self.cfg.DEVICE)
            
            # Forward pass
            preds = model(imgs)
            
            # Compute loss
            loss, _ = compute_loss(preds[1], targets)
            loss.backward()
