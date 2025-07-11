from abc import ABC, abstractmethod

from infra.utils.model_utils import ModelUtils

class PruningTree(ABC):
    """Defines an interface for the PruningTree types.
    """
    @abstractmethod
    def __init__(self, model_utils: ModelUtils):
        self.post_prune_listeners = [
            model_utils.build_dependency_graph,
            model_utils.build_module_name_mappings
        ]

    def call_post_prune_listeners(self):
        for fn in self.post_prune_listeners:
            fn()

    @abstractmethod
    def get_importance(self):
        pass

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
