import os
import asyncio
from typing import List, Dict, Any
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_ollama import OllamaLLM
from ollama_strategy.repository.ollama_strategy_repository import OllamaStrategyRepository
from langchain_text_splitters import TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

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
        chunks = text.split('[')
        return [chunk.replace(']', '').strip() for chunk in chunks if chunk.strip()]

class OllamaStrategyRepositoryImpl(OllamaStrategyRepository):
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

    async def fetch_growth_strategy(self, content_categories, ages, genders, visibility, platforms, investment_amount,
                                    upload_frequency, interested_influencer, userToken, request_id):
        prompt = PromptTemplate.from_template("""
                         user_input :( 컨텐츠 : {content_categories} 
                         나이대: {ages} 
                         성별: {genders} 
                         공개 정도: {visibility} 
                         플랫폼: {platforms} 
                         투자 가능 금액: {investment_amount} 
                         업로드 가능 주기: {upload_frequency} 
                         관심 있는 인플루언서: {interested_influencer}) 

                         context: {context}
                         question: {question}
                         """)
        try:
            # 현재 작업 디렉터리 경로를 얻기
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, "data", "influencer-feature.txt")

            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            text_splitter = BracketSplitter()
            split_documents = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Ollama 모델 설정
            llm = OllamaLLM(
                model="gemma2:2b",
                temperature=0,
                callbacks=[MyCustomAsyncHandler()],
                base_url="http://localhost:11434"
            )

            question = ("chain을 실행하면서 ref에 대한 모든 세부 내용들에 대해 단계 별로 생각하고 생각한 것에 대해 그대로 답변을 해줘. 구체적일수록 좋아"
                        "예를 들어 '~~가 꼭 알아야 할 ~~꿀팁' 타이틀의 컨텐츠 내용은 어떻게 구성해야 하는지를 알려주는 방식으로 해줘."
                        "위의 방식으로 각 소제목 마다 전략을 5가지 제시해줘."
                        "그리고 각 구체적인 내용들에 대해서는 bullet point말고 숫자로만 구분해줘.")

            # 검색 결과를 context로 변환하는 함수
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # 체인 재구성
            chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "content_categories": lambda x: content_categories,
                    "ages": lambda x: ages,
                    "genders": lambda x: genders,
                    "visibility": lambda x: visibility,
                    "platforms": lambda x: platforms,
                    "investment_amount": lambda x: investment_amount,
                    "upload_frequency": lambda x: upload_frequency,
                    "interested_influencer": lambda x: interested_influencer,
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            print(investment_amount, upload_frequency, interested_influencer)
            strategy = chain.invoke(question)
            print(strategy)
            return {"generatedText": strategy, "userToken": userToken, "request_id": request_id}

        except Exception as e:
            print(f"Error details: {str(e)}")
            return f"오류 발생: {e}"