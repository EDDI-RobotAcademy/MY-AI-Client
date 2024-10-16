import os
import sys


from first_user_defined_function_domain.service.fudf_service_impl import FudfServiceImpl
from first_user_defined_function_domain.service.request.fudf_just_for_test_request import FudfJustForTestRequest
from first_user_defined_function_domain.service.response.fudf_just_for_test_response import FudfJustForTestResponse
from growth_strategy.service.growth_strategy_service_impl import GrowthStrategyServiceImpl
from growth_strategy.service.request.growth_strategy_request import GrowthStrategyRequest
from growth_strategy.service.response.growth_strategy_response import GrowthStrategyResponse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))

from template.custom_protocol.service.custom_protocol_service_impl import CustomProtocolServiceImpl
from template.request_generator.request_class_map import RequestClassMap
from template.response_generator.response_class_map import ResponseClassMap

from user_defined_protocol.protocol import UserDefinedProtocolNumber


class UserDefinedProtocolRegister:

    @staticmethod
    def registerDefaultUserDefinedProtocol():
        customProtocolService = CustomProtocolServiceImpl.getInstance()
        firstUserDefinedFunctionService = FudfServiceImpl.getInstance()

        requestClassMapInstance = RequestClassMap.getInstance()
        requestClassMapInstance.addRequestClass(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            FudfJustForTestRequest
        )

        responseClassMapInstance = ResponseClassMap.getInstance()
        responseClassMapInstance.addResponseClass(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            FudfJustForTestResponse
        )

        customProtocolService.registerCustomProtocol(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            firstUserDefinedFunctionService.justForTest
        )

	# 여러분의 사용자 정의형 프로토콜 등록 파트
    @staticmethod
    def registerGrowthStrategyProtocol():
        customProtocolService = CustomProtocolServiceImpl.getInstance()
        GrowthStrategyService = GrowthStrategyServiceImpl.getInstance()

		# 여러분들이 구성한 것 (프로토콜과 request 등록)
        requestClassMapInstance = RequestClassMap.getInstance()
        requestClassMapInstance.addRequestClass(
            UserDefinedProtocolNumber.MY_TEAM_GROWTH_STRATEGY,
            GrowthStrategyRequest
        )

		# 여러분들이 구성한 것 (프로토콜과 response 등록)
        responseClassMapInstance = ResponseClassMap.getInstance()
        responseClassMapInstance.addResponseClass(
            UserDefinedProtocolNumber.MY_TEAM_GROWTH_STRATEGY,
            GrowthStrategyResponse
        )

		# 여러분들이 구성한 것 (프로토콜과 구동할 함수 등록)
        customProtocolService.registerCustomProtocol(
            UserDefinedProtocolNumber.MY_TEAM_GROWTH_STRATEGY,
            GrowthStrategyService.generateGrowthStrategy
        )


	# 초기 구동에서 호출하는 부분
    @staticmethod
    def registerUserDefinedProtocol():
		# 디폴트 구성
        UserDefinedProtocolRegister.registerDefaultUserDefinedProtocol()
	    # 여러분의 사용자 정의형 프로토콜 등록 파트
        UserDefinedProtocolRegister.registerGrowthStrategyProtocol()