import logging
import os

from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from constants import SOURCE_DIRECTORY, CHROMA_SETTINGS, PERSIST_DIRECTORY
from langchain.document_loaders import PDFPlumberLoader, DirectoryLoader

from semantic_chunking_helper import SematicChunkingHelper

import torch
from transformers import AutoTokenizer



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


def ingest_docs_from_source_dir(source_dir=SOURCE_DIRECTORY):
    """Ingest docs from source dir and using semantic chunking to split docs"""
    logger.info("Ingest with model")

    model_name = "dangvantuan/vietnamese-embedding"
    source_dir = './.Docs.t/'
    # PyPDFDirectoryLoader, supports loading dpf files
    docs = DirectoryLoader(
        source_dir, glob="**/*.pdf", loader_cls=PDFPlumberLoader
    ).load()
    chunks = filter_complex_metadata(docs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
# Ensure truncation is applied when tokenizing
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def truncate_text(text, max_length=512):
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)    
    
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    semantic_chunking = SematicChunkingHelper(
        docs=chunks, embeddings=embeddings, buffer_size=2, breakpoint_threshold=50, add_start_index = True
    )
    Chroma.from_texts(
        texts=semantic_chunking.text_chunks,
        embedding=embeddings,
        client_settings=CHROMA_SETTINGS,
        persist_directory=PERSIST_DIRECTORY + "_" + model_name,
    )


if __name__ == '__main__':
    ingest_docs_from_source_dir()
