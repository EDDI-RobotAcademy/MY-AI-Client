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

    async def fetch_growth_strategy(self, gender, age_group, mbti, topic, strength, reveal, platform,
                                    interested_influencer):
        prompt = PromptTemplate.from_template("""
                          Input :(
                          성별: {gender}
                          나이대: {age_group}
                          MBTI: {mbti}
                          성장하고 싶은 주제: {topic}
                          장점: {strength}
                          공개정도: {reveal}
                          활동하고 싶은 플랫폼: {platform}
                          관심있는 인플루언서: {interested_influencer})

                       Ref:(
               ## 20대 여성 ENFP 인플루언서 성장 전략 🎤✨
                **1. 입력 요약:** 20대 여성, ENFP 성격의 사용자는 노래를 좋아하고 창의력이 풍부하며, 얼굴을 공개하지 않고 인스타그램에서 노래 커버나 오리지널곡을 업로드하며 인플루언서를 꿈꿉니다. 버블디아를 좋아하며 노래가 실력 있는 인플루언서가 되고 싶어 합니다. 
                **2. 성향 분석:** ✨ **ENFP의 장점:** * 🌱 **아이디어 폭발!** 끊임없이 새로운 공감을 얻을 수 있는 노래 커버 아이디어, 팬들과 소통할 수 있는 챌린지, 재미있는 브이로그 컨셉 등을 만들어낼 수 있습니다. * 💖 **따뜻한 매력!** 진심을 담은 노래를 통해 다른 사람들에게 감정을 전달하고, 진솔한 소통으로 팬과의 깊은 유대감을 형성할 수 있습니다. * 😄 **밝고 활발한 에너지!** 댓글과 메시지에 적극적으로 응답하며 팬들과 소통하는 즐거움을 느낄 수 있습니다. ⛔ **ENFP의 단점:** * ⏳ **집중력 약점:** 계획적인 콘텐츠 제작과 일정한 업로드 스케쥴 관리를 위해 노력해야 합니다. * 🧘‍♀️ **완벽주의에 좌절!**: 모든 것을 완벽하게하려는 경향은 부담감으로 이어질 수 있습니다. 자유로운 표현 생각하고 즐거운 운영이 중요합니다! **전략:** ENFP 특성을 활용하여 창의적인 콘텐츠를 만들고, 꾸준함을 지향한다면 충분히 극복하실 수 있습니다. 즉정적인 산출보다는 장기적인 관점에서 계획을 세우고, 작은 목표 달성을 통해 자신감을 얻으시는 것이 도움이 됩니다. 
                **3. 인플루언서 분석:** 🌟 **버블디아**: 자신만의 독특한 음색과 감성적인 타이틀곡이 성공 요인! * **추천 전략:** 버블디아처럼 독특한 음색이나 스타일을 찾아 진솔한 감성을 담은 노래 제작을 추천합니다. 
                **4. 콘텐츠 전략:** 💡 **바로 시작! 당신만의 색깔을 드러내세요!**: * 🎵 **베이직 커버곡:** 좋아하는 노래 커버를 솔직하고 진솔하게 노래하세요. (ENFP의 감성과 공감대 형성에 효과적!) * 🎶 **오리지널곡:** 직접 작사 작곡한 곡을 발표하여 독창적인 매력을 보여주세요. (ENFP의 창의력을 뽐내는 좋은 기회!) * 📽️ **일상 브이로그:** 노래 연습 부분, 곡 완성 확인 과정 등을 꺼내 보세요. 팬들은 "진솔함" 게을 거부감 없이 받아들일 겁니다! * 👥 **SNS 챌린지 참여:** 다른 플랫폼 유저들과 참여하며 범위를 넓혀 보세요. 재미있는 트렌드를 따라가세요! * 💬 **라이브 방송:** 🎤 곡 한곡 소개, 질의응답, 노래 요청까지! 팬들과 소통하며 친밀감을 높이는 일하기 좋습니다.
                **5. 총정리: 성장 로드맵 🗺️
                1️ **DIY! 개성 도출 (1-2개월):** 자신만의 독특한 음색, 커버 스타일, 혹은 오리지널 개성을 찾고 성장! 2️ **꾸준함! 주문형 콘텐츠 제작 (지속적):** 일정 시간 간격으로 새로운 콘텐츠를 업로드하여 팬들의 기대감을 유지하세요. 3️ **진솔한 소통 속에서 성장 (지속적):** 댓글, 메시지에 꾸준히 응답하며 팬들과 대화하고 공감대를 형성하세요! 4️ 🎉 **새 계층에게 도달하기! 협업! 2-6개월:** 다른 크리에이터들과 협업하여 새로운 팬층에게 도달하세요. 5️ 📸 **인스타그램 활용 극대화 (주 2~3번):** 스토리 기능, 하이라이트 등을 활용하여 팬들과 소통하며 알림 최대화를 유지! **응원 메시지:** 🙌 지금부터 당신의 음악이 많은 사람들에게 힘이 되고, 즐거움을 선사할 수 있도록 응원합니다! ✨
                       )

                          #Question: {question}

                          #Answer:
                          1. 입력 요약 : 
                           (사용자 Input의 value를 요약해서 유저에게 설명해주세요.)

                          2. 성향 분석 : 
                           (사용자의 성별, 나이대, MBTI성향을 분석하여 해당 성향의 장점과 단점을 알려주고
                            인플루언서로서 활동할 때 장점을 극대화 하고 단점을 극복할 수 있는 전략을 세워주세요.)

                          3. 인플루언서 분석 : 
                           (#Context: {context}를 참고해서 사용자가 관심있는 인플루언서를 소개하고, 
                           이들이 어떤 컨텐츠를 통해 성공할 수 있었는지 참고하여
                           사용자의 성향에 적절한 맞춤전략을 세워주세요.)

                          4. 컨텐츠 전략 : 
                           (사용자가 원하는 주제와 장점, 본인의 공개정도(얼굴, 목소리 등 공개여부), 활동하고 싶은 플랫폼 정보를 바탕으로
                            어떠한 컨텐츠를 업로드하는 것이 성장에 도움이 될지 몇가지 구체적인 예시를 들어 전략을 세워주세요.)

                          5. 총정리 :
                           (위에서 언급한 성장 전략들을 총정리하여 간단한 로드맵을 작성해주세요.
                            마지막으로 인플루언서로서 성장하고자 하는 사용자를 응원하는 메시지를 출력해주세요.) 
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
            question = "Ref의 format을 참조해서 해당 인플루언서가 효과적으로 성장할 수 있는 전략을 제시해 주세요."
            # 사용자 입력 변수를 넣으려면 lambda _: 의 형태로 받아줘야함
            chain = (
                    {"context": retriever, "question": lambda _: question,
                     "gender": lambda _: gender, "age_group": lambda _: age_group,
                     "mbti": lambda _: mbti, "topic": lambda _: topic,
                     "strength": lambda _: strength, "reveal": lambda _: reveal,
                     "platform": lambda _: platform, "interested_influencer": lambda _: interested_influencer
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