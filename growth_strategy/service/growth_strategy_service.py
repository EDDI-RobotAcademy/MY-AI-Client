from abc import ABC, abstractmethod


class GrowthStrategyService(ABC):
    @abstractmethod
    def generateGrowthStrategy(self, *args):
        pass