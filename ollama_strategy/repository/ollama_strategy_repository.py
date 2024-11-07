from abc import ABC, abstractmethod


class OllamaStrategyRepository(ABC):
    @abstractmethod
    def fetch_growth_strategy(self, content_categories, ages, genders,
                visibility, platforms, investment_amount, upload_frequency, interested_influencer, userToken, request_id):
        pass