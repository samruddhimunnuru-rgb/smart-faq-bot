"""
ingest.py
---------
Creates the Chroma vector database for the Smart FAQ Bot.

Run from the project root:

    python -m src.ingest

Run this again whenever documents inside:

    data/schemes/

are added, removed, or modified.
"""

import sys
import shutil
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import (
    Chroma,
)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

from langdetect import detect


# ============================================================
# PROJECT CONFIG
# ============================================================

from src.config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    SUPPORTED_LANGUAGES,
)


# ============================================================
# EXACT PDF FILENAMES → DISPLAY NAMES
# ============================================================

SCHEME_DISPLAY_NAMES = {

    "Ayushman-English.pdf":
        "Ayushman Bharat (PM-JAY)",

    "Fasal_bhima_yojana-English.pdf":
        "Pradhan Mantri Fasal Bima Yojana (PMFBY)",

    "Free_Bus_yojana-kannada.pdf":
        "Shakti Yojana – Free Bus Scheme",

    "Gruha_jyothi-Kannada.pdf":
        "Gruha Jyothi Scheme",

    "Minority affiars-Hindi.pdf":
        "Ministry of Minority Affairs – Detailed Demands for Grants",

    "Mission_Shakti-English.pdf":
        "Mission Shakti – Women Empowerment Programme",

    "NSP-English.pdf":
        "National Scholarship for Postgraduate Studies",

    "PM AJAY-English.pdf":
        "Pradhan Mantri Anusuchit Jaati Abhyuday Yojana (PM-AJAY)",

    "PM_AJAY-English.pdf":
        "Pradhan Mantri Anusuchit Jaati Abhyuday Yojana (PM-AJAY)",

    "PM POSHAN SCHEME-English.pdf":
        "PM POSHAN – Pradhan Mantri Poshan Shakti Nirman",

    "Pm_sacanidhi-English.pdf":
        "PM SVANidhi – Street Vendor's AtmaNirbhar Nidhi",

    "PM-KISAN-English.pdf":
        "PM-KISAN – Pradhan Mantri Kisan Samman Nidhi",
}


# ============================================================
# GET DISPLAY NAME
# ============================================================

def scheme_name_from_filename(filename: str) -> str:

    # First try exact filename
    if filename in SCHEME_DISPLAY_NAMES:

        return SCHEME_DISPLAY_NAMES[filename]


    # Fallback for future documents
    stem = Path(filename).stem

    return (
        stem
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ============================================================
# DOCUMENT LANGUAGE DETECTION
# ============================================================

def detect_document_language(text: str) -> str:

    if not text or not text.strip():

        return "unknown"


    try:

        detected = detect(text)

        if detected in SUPPORTED_LANGUAGES:

            return detected

    except Exception:

        pass


    return "unknown"


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def load_all_documents():

    all_docs = []


    # --------------------------------------------------------
    # CHECK DATA DIRECTORY
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            "\nScheme folder not found:\n"
            f"{DATA_DIR}\n\n"
            "Create the folder and add PDF/TXT files."
        )


    # --------------------------------------------------------
    # FIND SUPPORTED FILES
    # --------------------------------------------------------

    files = sorted(
        DATA_DIR.glob("*")
    )


    supported_files = [

        file_path

        for file_path in files

        if file_path.suffix.lower()
        in (".pdf", ".txt")

    ]


    if not supported_files:

        raise FileNotFoundError(
            "\nNo PDF or TXT files found in:\n"
            f"{DATA_DIR}\n\n"
            "Add government scheme documents first."
        )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SMART FAQ BOT — DOCUMENT INGESTION")
    print("=" * 70)

    print()
    print(
        f"Found {len(supported_files)} document(s)"
    )

    print()


    # ========================================================
    # LOAD EACH DOCUMENT
    # ========================================================

    for file_path in supported_files:

        print("-" * 70)

        print(
            f"File: {file_path.name}"
        )


        # ----------------------------------------------------
        # DISPLAY NAME
        # ----------------------------------------------------

        scheme_name = (
            scheme_name_from_filename(
                file_path.name
            )
        )


        print(
            f"Display name: {scheme_name}"
        )


        # ----------------------------------------------------
        # LOAD PDF
        # ----------------------------------------------------

        try:

            if file_path.suffix.lower() == ".pdf":

                loader = PyPDFLoader(
                    str(file_path)
                )

            else:

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )


            docs = loader.load()


        except Exception as e:

            print()
            print(
                f"ERROR loading {file_path.name}:"
            )

            print(e)

            print(
                "Skipping this document..."
            )

            continue


        print(
            f"Pages/records loaded: {len(docs)}"
        )


        # ----------------------------------------------------
        # ADD METADATA
        # ----------------------------------------------------

        non_empty_pages = 0


        for doc in docs:

            # Detect language
            language = (
                detect_document_language(
                    doc.page_content
                )
            )


            # ------------------------------------------------
            # SCHEME NAME
            # ------------------------------------------------

            doc.metadata["scheme"] = (
                scheme_name
            )


            # ------------------------------------------------
            # ORIGINAL FILE NAME
            # ------------------------------------------------

            doc.metadata["source_file"] = (
                file_path.name
            )


            # ------------------------------------------------
            # LANGUAGE
            # ------------------------------------------------

            doc.metadata["language"] = (
                language
            )


            # ------------------------------------------------
            # PAGE NUMBER
            # ------------------------------------------------

            if "page" in doc.metadata:

                doc.metadata["page_number"] = (
                    doc.metadata["page"] + 1
                )


            # ------------------------------------------------
            # CHECK WHETHER TEXT WAS EXTRACTED
            # ------------------------------------------------

            if (
                doc.page_content
                and doc.page_content.strip()
            ):

                non_empty_pages += 1


        print(
            f"Non-empty pages/records: "
            f"{non_empty_pages}"
        )


        # ----------------------------------------------------
        # WARNING FOR SCANNED PDF
        # ----------------------------------------------------

        if len(docs) > 0 and non_empty_pages == 0:

            print()
            print(
                "WARNING:"
            )

            print(
                "No text could be extracted "
                "from this document."
            )

            print(
                "It may be a scanned/image PDF "
                "and may require OCR."
            )


        # ----------------------------------------------------
        # ADD TO ALL DOCUMENTS
        # ----------------------------------------------------

        all_docs.extend(
            docs
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 70)

    print(
        f"Total pages/records loaded: "
        f"{len(all_docs)}"
    )

    return all_docs


# ============================================================
# SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def chunk_documents(documents):

    print()
    print("=" * 70)
    print("STEP 2: SPLITTING DOCUMENTS INTO CHUNKS")
    print("=" * 70)


    # --------------------------------------------------------
    # TEXT SPLITTER
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


    # --------------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------------

    chunks = splitter.split_documents(
        documents
    )


    print()
    print(
        f"Created {len(chunks)} chunks."
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )


    return chunks


# ============================================================
# BUILD CHROMA VECTOR DATABASE
# ============================================================

def build_vectorstore(chunks):

    print()
    print("=" * 70)
    print("STEP 3: CREATING VECTOR DATABASE")
    print("=" * 70)


    # ========================================================
    # CHECK CHUNKS BEFORE DOING ANYTHING
    # ========================================================

    if not chunks:

        raise RuntimeError(

            "\nERROR: No text chunks were created "
            "from the documents.\n\n"

            "This usually means that the PDFs "
            "contain scanned images or no extractable text.\n\n"

            "Please check the PDF files."
        )


    # --------------------------------------------------------
    # DEBUG: NUMBER OF CHUNKS
    # --------------------------------------------------------

    print()

    print(
        f"DEBUG: Number of chunks to store: "
        f"{len(chunks)}"
    )


    # --------------------------------------------------------
    # DEBUG: FIRST CHUNK
    # --------------------------------------------------------

    print()

    print(
        "DEBUG: First chunk preview:"
    )

    print(
        chunks[0].page_content[:500]
    )


    # --------------------------------------------------------
    # DEBUG: FIRST METADATA
    # --------------------------------------------------------

    print()

    print(
        "DEBUG: First chunk metadata:"
    )

    print(
        chunks[0].metadata
    )


    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    print()

    print(
        "Loading embedding model:"
    )

    print(
        EMBEDDING_MODEL_NAME
    )


    print()

    print(
        "First run may download the embedding model."
    )

    print(
        "Please wait until ingestion finishes."
    )


    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL_NAME,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        },
    )


    # ========================================================
    # REMOVE OLD VECTOR DATABASE
    # ========================================================

    if VECTORSTORE_DIR.exists():

        print()

        print(
            "Removing old vector database..."
        )


        shutil.rmtree(
            VECTORSTORE_DIR
        )


    # --------------------------------------------------------
    # CREATE DIRECTORY
    # --------------------------------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CREATE CHROMA
    # ========================================================

    print()

    print(
        f"Embedding {len(chunks)} chunks..."
    )


    print()

    print(
        "Creating Chroma database..."
    )


    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(
            VECTORSTORE_DIR
        ),

        collection_name="government_schemes",
    )


    # ========================================================
    # VERIFY DATABASE
    # ========================================================

    print()

    print(
        "Verifying Chroma database..."
    )


    try:

        test_db = Chroma(

            collection_name="government_schemes",

            persist_directory=str(
                VECTORSTORE_DIR
            ),

            embedding_function=embeddings,
        )


        count = (
            test_db._collection.count()
        )


        print()

        print(
            f"DEBUG: Chroma stored chunks: "
            f"{count}"
        )


        if count == 0:

            raise RuntimeError(
                "Chroma database was created "
                "but contains 0 chunks."
            )


    except Exception as e:

        print()

        print(
            "WARNING: Could not verify "
            "Chroma database."
        )

        print(e)


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CHROMA DATABASE CREATED SUCCESSFULLY"
    )

    print("=" * 70)

    print()

    print(
        f"Vector database location:"
    )

    print(
        VECTORSTORE_DIR
    )

    print()

    print(
        f"Collection name:"
    )

    print(
        "government_schemes"
    )

    print()

    print(
        f"Chunks stored:"
    )

    print(
        len(chunks)
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SMART FAQ BOT")
    print("MULTILINGUAL DOCUMENT INGESTION")
    print("=" * 70)


    # ========================================================
    # STEP 1 — LOAD DOCUMENTS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1: LOADING GOVERNMENT SCHEME DOCUMENTS")
    print("=" * 70)


    documents = (
        load_all_documents()
    )


    if not documents:

        raise RuntimeError(
            "\nNo documents were successfully loaded."
        )


    print()

    print(
        f"Total pages/documents loaded: "
        f"{len(documents)}"
    )


    # ========================================================
    # STEP 2 — CHUNK DOCUMENTS
    # ========================================================

    chunks = (
        chunk_documents(
            documents
        )
    )


    # ========================================================
    # STEP 3 — BUILD VECTOR DATABASE
    # ========================================================

    build_vectorstore(
        chunks
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()

    print("=" * 70)

    print(
        "MULTILINGUAL INGESTION COMPLETE!"
    )

    print("=" * 70)

    print()

    print(
        "Vector database:"
    )

    print(
        VECTORSTORE_DIR
    )

    print()

    print(
        "Supported languages:"
    )


    for code, name in (
        SUPPORTED_LANGUAGES.items()
    ):

        print(
            f"  {code} → {name}"
        )


    print()

    print(
        "Next command:"
    )

    print(
        "streamlit run app.py"
    )

    print()

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()