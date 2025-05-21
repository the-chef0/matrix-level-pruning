from ckatorch.core import cka_batch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseModelUtils:
    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.module_to_name = {}
        for name, module in self.base_model.named_modules():
            self.module_to_name[module] = name
        self.name_to_module = {v: k for k, v in self.module_to_name.items()}
        self.module_name_to_similarity = {name: [] for name in self.module_to_name.values()}

    def extract_hidden(self, hidden: Tensor):
        if isinstance(hidden, Tensor):
            return hidden
        elif isinstance(hidden, tuple):
            return hidden[0]
        else:
            raise TypeError(f"Unknown type: {type(hidden)}")

    def hook_fn(self, module, input, output):
        input = self.extract_hidden(input)
        output = self.extract_hidden(output)
        curr_module_name = self.module_to_name[module]
        similarity_values = cka_batch(input, output).detach().cpu()
        self.module_name_to_similarity[curr_module_name].append(similarity_values)
