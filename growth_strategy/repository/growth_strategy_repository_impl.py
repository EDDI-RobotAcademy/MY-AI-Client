import os
import openai
import asyncio

from typing import List, Dict, Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from growth_strategy.repository.growth_strategy_repository import GrowthStrategyRepository
from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API Key가 준비되어 있지 않습니다!")

os.environ["OPENAI_API_KEY"] = openai.api_key
environ = os.getenv("ENV")

class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.1)
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.1)
        print("Hi! I just woke up. Your llm is ending")


class BracketSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        # '[' 기준으로 먼저 split하고, 각 부분에서 ']'를 제거합니다.
        chunks = text.split('[')
        return [chunk.replace(']', '').strip() for chunk in chunks if chunk.strip()]
class GrowthStrategyRepositoryImpl(GrowthStrategyRepository):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def fetch_growth_strategy(self, content_categories, ages, genders, visibility, platforms, investment_amount, upload_frequency, interested_influencer):

        prompt = PromptTemplate.from_template("""
                         user_input :( 컨텐츠 : {content_categories} 
                         나이대: {ages} 
                         성별: {genders} 
                         공개 정도: {visibility} 
                         플랫폼: {platforms} 
                         투자 가능 금액: {investment_amount} 
                         업로드 가능 주기: {upload_frequency} 
                         관심 있는 인플루언서: {interested_influencer}) 
                         
                         ref: 
                         ### [1]. 입력 요약 - {{컨텐츠}}: 뷰티 - {{나이대}}: 10대 - {{성별}}: 남자 - {{공개 정도}}: 얼굴 공개 - {{플랫폼}}: YouTube - {{투자 가능 금액}}: 없다 - {{업로드 가능 주기}}: 매일 - {{관심 있는 인플루언서}}: 레오제이 
                         ### [2]. 입력 분석 - 장점: - {{나이대}} {{성별}}이 {{컨텐츠}}에 도전하는 것은 비교적 유니크한 포지셔닝입니다. 기존의 여성 중심 {{콘텐츠}} 시장에서 새로운 트렌드와 다양성을 보여줄 수 있습니다. - {{공개 정도}}로 시청자와 더 깊이 소통할 수 있으며, 친근하고 신뢰감 있는 이미지를 구축할 수 있습니다. - {{업로드 가능 주기}} 업로드하는 일정한 콘텐츠 흐름은 {{플랫폼}} 알고리즘 상 노출 빈도를 높이고, 구독자와의 일관된 소통에도 유리합니다. - 단점: - {{콘텐츠}}에 대한 고정관념으로 인한 부정적인 반응이 있을 수 있으나, 자신감 있는 태도가 필요합니다. - {{투자 가능 금액}}의 상황에서는 고가의 제품 대신 학생들에게 맞는 기초 스킨케어와 저렴한 화장품 리뷰로 시작해야 합니다. - {{업로드 가능 주기}} 업로드는 체력적으로 부담이 될 수 있으니, 숏폼 콘텐츠를 활용해 효율성을 높입니다. --- 
                         ### [3]. 인플루언서 분석: {{관심 있는 인플루언서}} - {{관심 있는 인플루언서}}는 {{콘텐츠}} 유튜버로서 재치 있는 진행과 카리스마 있는 메이크업 스타일로 인기를 끌고 있습니다. - 다양한 {{콘텐츠}} 콘텐츠(메이크업 튜토리얼, 일상 브이로그)를 통해 시청자와 친근하게 소통하며, 자신의 스타일을 확고하게 유지한 것이 성공 요인입니다. 적용 전략: - {{관심 있는 인플루언서}}처럼 캐릭터와 자신감 있는 태도를 유지하며 시청자와 소통하세요. - 초기에는 간단한 학생용 스킨케어 팁이나 저가 메이크업 튜토리얼로 시작하고, 점차 본인만의 {{콘텐츠}} 스타일을 찾아가는 것이 좋습니다. - {{관심 있는 인플루언서}}처럼 유머와 스토리텔링을 더해 시청자와 친밀감을 쌓는 것이 중요합니다. --- 
                         ### [4]. 콘텐츠 전략 1. 기초 스킨케어 및 저예산 화장품 리뷰 - 학생들에게 맞는 저렴한 스킨케어 제품 리뷰와 여드름 관리법을 소개합니다. - “10대가 꼭 알아야 할 {{콘텐츠}} 꿀팁”과 같은 타이틀로 시청자들의 관심을 유도합니다. 2. 데일리 메이크업 챌린지 - 일주일 동안 매일 다른 테마로 메이크업을 시도하는 챌린지 콘텐츠를 기획합니다. - 예: "학교 가기 전 5분 메이크업," "K-팝 아이돌 룩 도전하기." 3. {{콘텐츠}} 관련 브이로그 - 자신의 일상 속에서 화장품을 사용하는 모습을 담은 브이로그를 통해 친근한 이미지를 구축합니다. - "화장품 구매 브이로그" 또는 "메이크업 실습일기"와 같은 콘텐츠가 유효합니다. 4. 유머와 스토리텔링 활용 - {{관심 있는 인플루언서}}처럼 재미있는 일화나 에피소드를 곁들여 콘텐츠를 만듭니다. 시청자들이 쉽게 공감할 수 있도록 유머러스한 진행을 시도해보세요. --- 
                         ### [5]. 예산별 장비 및 툴 추천 - {{투자 가능 금액}}에 맞는 장비들을 추천합니다. - 예: "Adobe-Pro" (10만원) --- 
                         ### [6]. 총정리: 성장 로드맵 1. 1~2개월: 저예산 스킨케어 및 메이크업 제품 리뷰로 시작하여 구독자 기반을 다집니다. 2. 3~6개월: {{업로드 가능 주기}} 업로드를 유지하며, 다양한 메이크업 챌린지에 도전합니다. 3. 6개월~1년: 댓글과 라이브 방송을 활용해 소통을 강화하고, 새로운 트렌드와 협업 기회를 모색합니다. 4. 1년 후: 본인만의 {{콘텐츠}} 스타일을 확립하고, 다른 {{플랫폼}} 인플루언서와의 콜라보 콘텐츠로 새로운 팬층을 유입시킵니다. --- 
                         ### [7]. 응원 메시지 🎉 {{콘텐츠}} 유튜버의 길에 도전하는 당신을 응원합니다! {{관심 있는 인플루언서}}처럼 자신감 있는 태도로 새로운 {{콘텐츠}} 트렌드를 이끌어가세요. 꾸준한 노력과 진솔한 소통이 당신의 가장 큰 무기가 될 것입니다. 당신의 성장이 곧 1 {{나이대}} {{성별}} {{콘텐츠}} 시장의 새로운 기준이 될 거예요! 
                       """)
        try:
            # 단계 1: 문서 로드(Load Documents)
            # 현재 작업 디렉터리 경로를 얻기
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, "data", "influencer-feature.txt")

            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            # 단계 2: 문서 분할(Split Documents)
            text_splitter = BracketSplitter()
            split_documents = text_splitter.split_documents(docs)

            # 단계 3: 임베딩(Embedding) 생성
            embeddings = OpenAIEmbeddings()

            # 단계 4: DB 생성(Create DB) 및 저장
            vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

            # 단계 5: 검색기 생성
            retriever = vectorstore.as_retriever()

            # 단계 7: LLM 모델 생성 (call back함수 등록)
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, callbacks=[MyCustomAsyncHandler()])

            # 단계 8: 체인 생성
            question = ("chain을 실행하면서 ref에 대한 모든 세부 내용들에 대해 CoT 기법을 적용해서 더 구체적으로 답변을 해줘. "
                        "예를 들어 '~~가 꼭 알아야 할 ~~꿀팁' 타이틀의 컨텐츠 내용은 어떻게 구성해야 하는지를 알려주는 방식으로 해줘.")
            # 사용자 입력 변수를 넣으려면 lambda _: 의 형태로 받아줘야함
            chain = (
                    {"context": retriever, "question": lambda _: question,
                     "genders": lambda _: genders, "ages": lambda _: ages,
                     "content_categories": lambda _: content_categories, "upload_frequency": lambda _: upload_frequency,
                     "investment_amount": lambda _: investment_amount, "visibility": lambda _: visibility,
                     "platforms": lambda _: platforms, "interested_influencer": lambda _: interested_influencer
                     }
                    | prompt
                    | llm
                    | StrOutputParser()
            )
            # 단계 9: 체인 실행
            strategy = chain.invoke(question) # 여기서 에러 터짐
            print(strategy)
            return {"generatedText": strategy}  # dict 형식으로 반환해주어야 함
        except Exception as e:
            return f"오류 발생: {e}"