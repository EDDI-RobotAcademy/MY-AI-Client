import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))

from template.custom_protocol.service.custom_protocol_service_impl import CustomProtocolServiceImpl
from template.request_generator.request_class_map import RequestClassMap
from template.response_generator.response_class_map import ResponseClassMap

from user_defined_protocol.protocol import UserDefinedProtocolNumber


class UserDefinedProtocolRegister:
    # 우리가 사용하는 모델에 맞게 나중에 이름 수정 필요
    @staticmethod
    def registerOpenaiApiProtocol():
        customProtocolService = CustomProtocolServiceImpl.getInstance()
        openaiApiService = OpenaiApiServiceImpl.getInstance()

        requestClassMapInstance = RequestClassMap.getInstance()
        requestClassMapInstance.addRequestClass(
            UserDefinedProtocolNumber.OPENAI_API,
            OpenaiApiRequest
        )

        responseClassMapInstance = ResponseClassMap.getInstance()
        responseClassMapInstance.addResponseClass(
            UserDefinedProtocolNumber.OPENAI_API,
            OpenaiApiResponse
        )

        customProtocolService.registerCustomProtocol(
            UserDefinedProtocolNumber.OPENAI_API,
            openaiApiService.letsChat
        )



    @staticmethod
    def registerUserDefinedProtocol():
        UserDefinedProtocolRegister.registerOpenaiApiProtocol()

