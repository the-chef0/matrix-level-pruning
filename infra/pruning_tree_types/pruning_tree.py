from abc import ABC, abstractmethod

class PruningTree(ABC):
    """Defines an interface for the PruningTree types.
    """

    @abstractmethod
    def get_importance(self):
        pass

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
