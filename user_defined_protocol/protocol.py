from enum import Enum


class UserDefinedProtocolNumber(Enum):
    # 예약된 정보 (1, 2, 11, 12, 13, 21) 을 제외하고 사용하도록 함

    # 추가된 프로토콜 for growth_strategy test
    MY_TEAM_GROWTH_STRATEGY = 7
    MY_TEAM_OLLAMA_STRATEGY = 8
    # 기존에 있던 테스트용 프로토콜
    FIRST_USER_DEFINED_FUNCTION_FOR_TEST = 5

    @classmethod
    def hasValue(cls, value):
        return any(value == item.value for item in cls)
