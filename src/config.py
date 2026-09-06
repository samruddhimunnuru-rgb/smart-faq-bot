"""
config.py
---------
Central configuration for Smart FAQ Bot.

Architecture:
- Ollama -> local LLM
- HuggingFace -> multilingual embeddings
- Chroma -> vector database
- SQLite -> recent query history
- No Gemini/OpenAI API required
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlsplit

# ENVIRONMENT

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8501"))


# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

RUNTIME_DIR = Path(
    os.getenv(
        "SMART_FAQ_RUNTIME_DIR",
        str(Path(tempfile.gettempdir()) / "smart-faq-bot-runtime"),
    )
)

SOURCE_DATA_DIR = BASE_DIR / "data" / "schemes"
DATA_DIR = RUNTIME_DIR / "data" / "schemes"

SOURCE_VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR = RUNTIME_DIR / "vectorstore"

SOURCE_REGISTRY_PATH = RUNTIME_DIR / "source_registry.json"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "scheme-documents")

# CHUNKING

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "700")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "100")
)


# SUPPORTED LANGUAGES

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
}

# MULTILINGUAL EMBEDDING MODEL

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2"
)

# RETRIEVAL

TOP_K = int(
    os.getenv("TOP_K", "4")
)

# OLLAMA

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

_ollama_base_url = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip().rstrip("/")
if _ollama_base_url and not urlsplit(_ollama_base_url).scheme:
    _ollama_base_url = f"https://{_ollama_base_url}"
OLLAMA_BASE_URL = _ollama_base_url

# PUBLIC DEPLOYMENT / CLOUD MODE
# If a hosted Ollama endpoint is supplied, this app will use it automatically.
# Example: OLLAMA_BASE_URL=https://your-ollama-service.example.com


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are "Smart FAQ Bot", a multilingual assistant that helps
citizens understand Indian government schemes.

You are a DOCUMENT-GROUNDED RAG assistant.

Follow these rules STRICTLY:

1. Answer ONLY from the information contained in the
   supplied CONTEXT.

2. Do NOT use outside knowledge.

3. Do NOT browse the internet.

4. The government document and the user's question may
   be written in DIFFERENT languages.

5. Understand the supplied context regardless of whether
   it is written in English, Hindi, Kannada, Tamil, Telugu,
   or Malayalam.

6. ALWAYS answer in the SAME LANGUAGE as the user's question.

7. The language of the SOURCE DOCUMENT must NOT determine
   the language of the answer.

8. If the user asks in Kannada and the information is found
   in a Hindi document, answer in Kannada.

9. If the user asks in Hindi and the information is found
   in an English document, answer in Hindi.

10. Do NOT translate the answer to English unless the
    user's question is in English.

11. Never invent:
    - eligibility criteria
    - required documents
    - benefits
    - benefit amounts
    - application procedures
    - dates
    - deadlines
    - government rules
    - scheme requirements

12. If the answer is not present in the context, clearly
    say that the approved documents do not contain the
    requested information.

13. Keep the answer simple and easy for a citizen to
    understand.

14. Keep the answer focused on the user's question.

15. If the context contains conflicting or unclear
    information, do not guess.

SUPPORTED QUESTION LANGUAGES:

English
Hindi
Kannada
Tamil
Telugu
Malayalam


SOURCE INFORMATION:
{source_information}


CONTEXT FROM APPROVED GOVERNMENT SCHEME DOCUMENTS:
{context}


USER QUESTION:
{question}


USER QUESTION LANGUAGE:
{question_language}


ANSWER:
"""

# FALLBACK MESSAGE

FALLBACK_MESSAGE = (
    "I don't have this information in my approved documents. "
    "Please check the official scheme website or visit the "
    "nearest Common Service Centre (CSC)."
)