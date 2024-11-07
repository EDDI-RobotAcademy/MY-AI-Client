from ollama_strategy.repository.ollama_strategy_repository_impl import OllamaStrategyRepositoryImpl
from ollama_strategy.service.ollama_strategy_service import OllamaStrategyService

class OllamaStrategyServiceImpl(OllamaStrategyService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__OllamaStrategyRepository = OllamaStrategyRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def generateOllamaStrategy(self, *args):
        try:
            print(f"OllamaStrategyService!")

            ages = args[0]
            genders = args[1]
            content_categories = args[2]
            visibility = args[3]
            platforms = args[4]
            investment_amount = args[5]
            upload_frequency = args[6]
            interested_influencer = args[7]

            userToken = args[8]
            request_id = args[9]

            return await self.__OllamaStrategyRepository.fetch_growth_strategy(content_categories, ages, genders,
            visibility, platforms, investment_amount, upload_frequency, interested_influencer, userToken, request_id)
        except Exception as e:
            print(e)
            raise Exception from e
