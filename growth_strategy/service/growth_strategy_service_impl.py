from growth_strategy.repository.growth_strategy_repository_impl import GrowthStrategyRepositoryImpl
from growth_strategy.service.growth_strategy_service import GrowthStrategyService

class GrowthStrategyServiceImpl(GrowthStrategyService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__GrowthStrategyRepository = GrowthStrategyRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def generateGrowthStrategy(self, userSendMessage):
        try:
            print(f"GrowthStrategyService!", userSendMessage)

            return await self.__GrowthStrategyRepository.fetch_growth_strategy(
            userSendMessage["gender"],
            userSendMessage["age_group"],
            userSendMessage["mbti"],
            userSendMessage["topic"],
            userSendMessage["strength"],
            userSendMessage["reveal"],
            userSendMessage["platform"],
            userSendMessage["interested_influencer"]
            )
        except Exception as e:
            print(e)
