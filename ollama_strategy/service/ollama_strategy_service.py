from abc import ABC, abstractmethod


class OllamaStrategyService(ABC):
    @abstractmethod
    def generateOllamaStrategy(self, *args):
        pass