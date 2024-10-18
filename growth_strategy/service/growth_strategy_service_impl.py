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

    async def generateGrowthStrategy(self, *args):
        try:
            print(f"GrowthStrategyService!")
            gender = args[0]
            age_group = args[1]
            mbti = args[2]
            topic = args[3]
            strength = args[4]
            reveal = args[5]
            platform = args[6]
            interested_influencer = args[7]

            return await self.__GrowthStrategyRepository.fetch_growth_strategy(
            gender, age_group, mbti, topic, strength, reveal, platform, interested_influencer
            )
        except Exception as e:
            print(e)
            raise Exception from e
