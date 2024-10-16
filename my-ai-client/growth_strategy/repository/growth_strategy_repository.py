from abc import ABC, abstractmethod

class GrowthStrategyRepository(ABC):

    @abstractmethod
    def fetch_growth_strategy(self,gender, age_group, mbti, topic, strength, reveal, platform, interested_influencer):
        pass