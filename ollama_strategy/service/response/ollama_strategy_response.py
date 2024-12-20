from user_defined_protocol.protocol import UserDefinedProtocolNumber


class OllamaStrategyResponse:
    def __init__(self, responseData):
		# 등록한 프로토콜 번호
        self.protocolNumber = UserDefinedProtocolNumber.MY_TEAM_OLLAMA_STRATEGY.value

        for key, value in responseData.items():
            setattr(self, key, value)

    @classmethod
    def fromResponse(cls, responseData):
        return cls(responseData)

    def toDictionary(self):
        return self.__dict__

    def __str__(self):
		# response 이름
        return f"OllamaStrategyResponse({self.__dict__})"