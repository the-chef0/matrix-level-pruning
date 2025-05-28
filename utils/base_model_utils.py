from torch import Tensor
from torch.nn import SiLU
from torch_pruning.dependency import DependencyGraph
from torch_pruning.pruner.function import BasePruningFunc, LayernormPruner
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm

def extract_logits(output: Tensor) -> Tensor:
    return output.logits

class ActivationPruner(BasePruningFunc):

    def prune_out_channels(self, layer, idxs):
        return layer
    
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer):
        return None
    
    def get_in_channels(self, layer):
        return None

class BaseModelUtils:
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pruning_excluded_keywords = ['self_attn', 'lm_head']

        self.dep_graph = None
        self.nonlinearities_forward = [SiLU] # TODO: Include all activations from TP
        self.nonlinearities_backward = [LlamaRMSNorm]

        self.module_to_name = None
        self.name_to_module = None

    def build_module_name_mappings(self):
        self.module_to_name = {}
        for name, module in self.base_model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}

    def build_dependency_graph(self):
        self.dep_graph = DependencyGraph().build_dependency(
            model=self.base_model,
            example_inputs=self.base_model.dummy_inputs['input_ids'].to('cuda'),
            output_transform=extract_logits,
            customized_pruners={LlamaRMSNorm: LayernormPruner(), SiLU: ActivationPruner()}
        )

    def extract_hidden(self, hidden: Tensor):
        if isinstance(hidden, Tensor):
            return hidden
        elif isinstance(hidden, tuple):
            return hidden[0]
        else:
            raise TypeError(f"Unknown type: {type(hidden)}")
