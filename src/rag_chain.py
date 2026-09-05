"""
rag_chain.py
------------
Multilingual RAG engine.

Flow:

User question
      ↓
Multilingual embedding
      ↓
Chroma retrieval
      ↓
Relevant document chunks
      ↓
Ollama
      ↓
Answer in user's language

The original question is NOT translated to English
before retrieval.
"""

import sys
from pathlib import Path


# PROJECT ROOT

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )



# LANGCHAIN

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import (
    Chroma,
)

from langchain_ollama import (
    ChatOllama,
)

# CONFIG

from src.config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
    FALLBACK_MESSAGE,
    SUPPORTED_LANGUAGES,
)

# GLOBAL OBJECTS


_embeddings = None

_vectorstore = None

_llm = None

_answer_cache = {}


def reset_runtime_cache():
    """Force the next question to reload the updated vector database."""
    global _embeddings
    global _vectorstore
    global _llm

    _embeddings = None
    _vectorstore = None
    _llm = None
    _answer_cache.clear()


# VECTORSTORE

def _get_vectorstore():

    global _embeddings
    global _vectorstore


    if _vectorstore is not None:
        return _vectorstore


    # CHECK DATABASE

    if (
        not VECTORSTORE_DIR.exists()
        or not any(
            VECTORSTORE_DIR.iterdir()
        )
    ):

        raise FileNotFoundError(
            "Vector database not found.\n\n"
            "Run this first:\n"
            "python -m src.ingest"
        )


    # LOAD EMBEDDING MODEL
  
    _embeddings = HuggingFaceEmbeddings(

        model_name=(
            EMBEDDING_MODEL_NAME
        ),

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    # LOAD CHROMA
    
    _vectorstore = Chroma(

        persist_directory=str(
            VECTORSTORE_DIR
        ),

        embedding_function=(
            _embeddings
        ),

        collection_name=(
            "government_schemes"
        ),
    )


    return _vectorstore

# OLLAMA


def _get_llm():

    global _llm


    if _llm is None:

        _llm = ChatOllama(

            model=OLLAMA_MODEL,

            base_url=OLLAMA_BASE_URL,

            temperature=0,

            num_ctx=4096,
        )


    return _llm


# RETRIEVE

def retrieve_chunks(
    question: str,
    k: int = TOP_K,
):

    vectorstore = (
        _get_vectorstore()
    )


    retriever = (
        vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={
                "k": k
            },
        )
    )



    return retriever.invoke(
        question
    )

# ANSWER FROM DOCUMENTS

def _answer_from_documents(
    question: str,
    question_language: str,
):

    chunks = retrieve_chunks(
        question
    )


    if not chunks:

        return (
            None,
            [],
            [],
        )


    context_parts = []

    sources = []


    # BUILD CONTEXT

    for chunk in chunks:

        scheme = chunk.metadata.get(
            "scheme",
            "Unknown Scheme",
        )

        source_file = chunk.metadata.get(
            "source_file",
            "",
        )

        source_url = chunk.metadata.get(
            "source_url",
            "",
        )

        document_language = (
            chunk.metadata.get(
                "language",
                "unknown",
            )
        )


        context_parts.append(
            f"""
SOURCE SCHEME:
{scheme}

SOURCE FILE:
{source_file}

OFFICIAL SOURCE URL:
{source_url or "Not recorded"}

SOURCE DOCUMENT LANGUAGE:
{document_language}

CONTENT:
{chunk.page_content}
"""
        )


        source_name = f"{scheme} ({source_file})" if source_file else scheme
        if source_url:
            source_name += f" - {source_url}"


        if source_name not in sources:

            sources.append(
                source_name
            )


    context = (
        "\n\n--------------------\n\n"
        .join(
            context_parts
        )
    )


    # LANGUAGE NAME

    language_name = (
        SUPPORTED_LANGUAGES.get(
            question_language,
            "English",
        )
    )


    # PROMPT

    prompt = SYSTEM_PROMPT.format(

        source_information=(
            ", ".join(sources)
        ),

        context=context,

        question=question,

        question_language=(
            language_name
        ),
    )


    # OLLAMA
   
    llm = _get_llm()


    response = llm.invoke(
        prompt
    )


    if hasattr(
        response,
        "content"
    ):

        answer = response.content

    else:

        answer = str(response)


    return (
        answer,
        sources,
        chunks,
    )

# MAIN ANSWER FUNCTION

def answer_question(
    question: str,
    question_language: str = "en",
):

    question = question.strip()


    if not question:

        return (
            FALLBACK_MESSAGE,
            [],
            [],
            False,
        )


    # CACHE

    cache_key = (
        question.lower(),
        question_language,
    )


    if cache_key in _answer_cache:

        return _answer_cache[
            cache_key
        ]

    # RAG

    try:

        (
            answer,
            sources,
            chunks,
        ) = _answer_from_documents(

            question,

            question_language,
        )


    except Exception as e:

        result = (
            f"⚠️ RAG error: {e}",
            [],
            [],
            False,
        )

        return result


    # NO RETRIEVED DOCUMENTS

    if not chunks:

        result = (
            FALLBACK_MESSAGE,
            [],
            [],
            False,
        )

        _answer_cache[
            cache_key
        ] = result

        return result

    # RESULT

    result = (
        answer,
        sources,
        chunks,
        False,
    )


    _answer_cache[
        cache_key
    ] = result


    return result

# TERMINAL TEST

if __name__ == "__main__":

    print(
        "\nSmart FAQ Bot"
    )

    print(
        "Multilingual RAG Test"
    )

    print(
        "Type 'quit' to exit.\n"
    )


    while True:

        question = input(
            "Ask a question: "
        ).strip()


        if question.lower() in (
            "quit",
            "exit",
        ):

            break


        if not question:

            continue


        print(
            "\nChecking documents...\n"
        )


        answer, sources, _, _ = (
            answer_question(
                question,
                "en",
            )
        )


        print(
            answer
        )


        print(
            "\nSources:"
        )


        if sources:

            for source in sources:

                print(
                    f"  - {source}"
                )

        else:

            print(
                "  None"
            )


        print(
            "\n" + "-" * 60
        )