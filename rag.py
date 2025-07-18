import logging
import os
import re
from datetime import datetime

from langchain.document_loaders import DirectoryLoader, JSONLoader, PDFPlumberLoader
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.chat_models import ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnableParallel

# from ingest import RAGIngest
from callback_logger import CallbackLogger
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, SOURCE_DIRECTORY, MODELS
from performance_logger import PerformanceLogger
from semantic_chunking_helper import SematicChunkingHelper

set_debug(True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)
performance_logger = PerformanceLogger()


def format_docs(docs):
    """Returns a string with all the page_content joined with double newlines."""
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
        logger.info("persistent_dir: %s", self.persist_dir)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Bạn là môt trợ lý ảo chuyên trả lời các câu hỏi liên quan đến lĩnh vực y tế và du lịch.\
            Đừng cung cấp cho tôi bất kỳ thông tin nào ngoài những thông tin được cung cấp. \
            Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết hoặc chưa đủ thông tin. \
            Hãy trả lời toàn bộ bằng Tiếng Việt [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

        self.suggestion_prompt = PromptTemplate.from_template(
            """
            <s> [INST] Dựa trên câu hỏi và câu trả lời sau đây, hãy đề xuất 3 câu hỏi tiếp theo mà người dùng có thể quan tâm. \
            Hãy đảm bảo các câu hỏi đề xuất liên quan đến chủ đề hiện tại và giúp người dùng khám phá thêm về vấn đề. \
            Trả lời bằng tiếng Việt và định dạng câu trả lời dưới dạng danh sách đánh số.  \
            Chỉ cần đưa ra những câu hỏi gợi ý, không giải thích gì thêm, không dịch nội dung qua tiếng Anh.  \
            Không đưa vào câu "Here are three potential follow-up questions that users may be interested in:" trong gợi ý. [/INST] </s>

            [INST] Câu hỏi: {question}
            Câu trả lời: {answer}

            Các câu hỏi đề xuất: [/INST]
            """
        )

    def ingest_docs_from_source_dir(self, source_dir=SOURCE_DIRECTORY):
        """Ingest docs from source dir"""
        logger.info("Ingest with model: %s", self.model_name)

        def medical_metadata_func(record: dict, metadata: dict) -> dict:
            metadata["title"] = record.get("title")
            metadata["link"] = record.get("link")
            metadata["topic"] = "y tế"
            # Extract subtopic
            file_source = metadata.get("source")
            metadata["subtopic"] = (
                os.path.basename(file_source).split(".")[0] if file_source else None
            )
            # Extract created_date
            date_str = re.search(
                r"\d{2}/\d{2}/\d{4}", record.get("created_date")
            ).group()
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            metadata["created_date"] = date_obj
            return metadata

        json_docs = DirectoryLoader(
            source_dir,
            glob="**/*.json",
            loader_cls=JSONLoader,
            loader_kwargs={
                "jq_schema": ".[]",
                "content_key": "content",
                "metadata_func": medical_metadata_func,
            },
            show_progress=True,
            use_multithreading=True,
        ).load()
        # print(json_docs[0].metadata)
        docs = DirectoryLoader(
            source_dir,
            glob="**/*.pdf",
            loader_cls=PDFPlumberLoader,
            show_progress=True,
            use_multithreading=True,
        ).load()
        if json_docs is not None:
            docs.extend(json_docs)
        chunks = filter_complex_metadata(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name=MODELS.get("embeddings"),
            model_kwargs={"device": MODELS.get("device")},  # use "mps" for Apple Silicon Chip
        )

        err_count = 0
        for index, chunk in enumerate(chunks):
            try:
                # Extract the topic, document name and page number
                topic = chunk.metadata.get("topic", "du lịch")
                doc_name = chunk.metadata.get(
                    "source", "Unknown Document"
                )  # Extract doc path/name
                page_number = chunk.metadata.get(
                    "page", "Unknown Page"
                )  # Extract page number

                # Add the document name and page number to the metadata
                chunk.metadata["topic"] = topic
                chunk.metadata["document_name"] = doc_name
                chunk.metadata["page_index"] = page_number

                # Perform semantic chunking and store chunks with metadata
                semantic_chunking = SematicChunkingHelper(
                    docs=[chunk],
                    embeddings=embeddings,
                    buffer_size=2,
                    breakpoint_threshold=50,
                )
                Chroma.from_texts(
                    texts=semantic_chunking.text_chunks,
                    embedding=embeddings,
                    metadatas=[
                        chunk.metadata for _ in semantic_chunking.text_chunks
                    ],  # Attach metadata
                    client_settings=CHROMA_SETTINGS,
                    persist_directory=self.persist_dir,
                )

            except Exception as e:
                err_count += 1
                logger.error(f"Cannot ingest page {index}: {str(e)}")

        if err_count > 0:
            logger.error(f"{err_count} pages not ingested")

    def load_retriever(self):
        if self.retriever is not None:
            return True

        if not os.path.exists(self.persist_dir):
            return False

        embeddings = HuggingFaceEmbeddings(
            model_name=MODELS.get("embeddings"),
            model_kwargs={"device": MODELS.get("device")},  # note for Apple Silicon Chip use "mps"
        )

        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )

        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker(model=model, top_n=2)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        self.chain = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=chain_from_docs)

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

    def ask(self, query: str):
        """Retrieve answer from LLM"""
        if not self.chain:
            return "Please, add a document first."

        performance_logger.open(query, model=self.model_name)
        callback_handler = CallbackLogger(logger=performance_logger)
        config = {"callbacks": [callback_handler]}

        result = self.chain.invoke(query, config=config)
        performance_logger.close(result)

        self.filter_answer_from_response(result)

        suggestions = self.generate_suggestions(query, result["answer"])

        # Add suggestions to the result
        result["suggestions"] = suggestions

        return result

    def generate_suggestions(self, question: str, answer: str):
        suggestion_chain = self.suggestion_prompt | self.model | StrOutputParser()
        suggestions = suggestion_chain.invoke({"question": question, "answer": answer})
        return suggestions.strip()

    def clear(self):
        """Clear"""
        self.vector_store = None
        self.retriever = None
        self.chain = None
