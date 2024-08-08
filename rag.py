import logging
import os

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, SOURCE_DIRECTORY
from callback_logger import CallbackLogger
from performance_logger import PerformanceLogger
from langchain.globals import set_debug
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from semantic_chunking_helper import SematicChunkingHelper
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.document_loaders import PDFPlumberLoader, DirectoryLoader

set_debug(True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
performance_logger = PerformanceLogger()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAG:
    """RAG Class Helper"""

    vector_store = None
    retriever = None
    chain = None
    persist_dir = None

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name)
        self.persist_dir = PERSIST_DIRECTORY + "_" + model_name

    def ingest_docs_from_source_dir(self, source_dir=SOURCE_DIRECTORY):
        """Ingest docs from source dir and using semantic chunking to split docs"""
        logger.info("Ingest with model: %s", self.model_name)
        # PyPDFDirectoryLoader, supports loading dpf files
        docs = DirectoryLoader(
            source_dir, glob="**/*.pdf", loader_cls=PDFPlumberLoader
        ).load()
        chunks = filter_complex_metadata(docs)
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
            model_kwargs={"device": "cpu"},  # note for Apple Silicon Chip use "mps"
        )
        semantic_chunking = SematicChunkingHelper(
            docs=chunks, embeddings=embeddings, buffer_size=2, breakpoint_threshold=50
        )
        Chroma.from_texts(
            texts=semantic_chunking.text_chunks,
            embedding=embeddings,
            client_settings=CHROMA_SETTINGS,
            persist_directory=self.persist_dir,
        )

    def create_conversational_chain(self, model: ChatOllama):
        """Create chat history"""
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                                            which might reference context in the chat history, formulate a standalone question \
                                            which can be understood without the chat history. Do NOT answer the question, \
                                            just reformulate it if needed and otherwise return it as is.
                                        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            model, self.retriever, contextualize_q_prompt
        )

        qa_system_prompt = """
                            You are a helpful DEK assistant for question-answering DEK policies. \
                            Do not give me any information outside of PROVIDED CONTEXT. \
                            If you don't know the answer, just say that you don't know. \
                            {context}
                            """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | qa_prompt
            | self.model
            | StrOutputParser()
        )

        question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        ).assign(answer=chain_from_docs)

        return rag_chain

    def load_retriever(self):
        if self.retriever is not None:
            return True

        if not os.path.exists(self.persist_dir):
            return False

        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
            model_kwargs={"device": "cpu"},  # note for Apple Silicon Chip use "mps"
        )

        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )

        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        model = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base", model_kwargs={"device": "mps"}
        )
        reranker = CrossEncoderReranker(model=model, top_n=2)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        self.chain = self.create_conversational_chain(self.model)

        return True

    def filter_answer_from_response(self, response: dict):
        """filter_answer_from_response"""
        if response["answer"] or len(response["answer"]) > 0:
            response["answer"] = response["answer"].replace("</s> [INST]", "")
            response["answer"] = response["answer"].replace("</s>", "")
            response["answer"] = response["answer"].replace("<s>", "")
            response["answer"] = response["answer"].replace("[ANSW]", "")
            response["answer"] = response["answer"].replace("[ANS]", "")
            response["answer"] = response["answer"].replace("[/ANSW]", "")
            response["answer"] = response["answer"].replace("[INST]", "")
            response["answer"] = response["answer"].replace("[/INST]", "")

    def ask(self, query: str, chat_history: list):
        """Retrieve answer from LLM"""
        if not self.chain:
            return "Please, add a document first."

        performance_logger.open(query, model=self.model_name)
        callback_handler = CallbackLogger(logger=performance_logger)
        config = {"callbacks": [callback_handler]}

        result = self.chain.invoke(
            {"input": query, "chat_history": chat_history}, config=config
        )
        performance_logger.close(result)

        self.filter_answer_from_response(result)

        return result

    def clear(self):
        """Clear"""
        self.vector_store = None
        self.retriever = None
        self.chain = None
