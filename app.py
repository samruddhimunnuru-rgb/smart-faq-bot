"""
app.py
------
Smart FAQ Bot — Multilingual Streamlit Frontend

Features:
- Text questions
- Voice questions
- Automatic language detection
- 6 supported Indian languages
- Cross-language document retrieval
- Same-language text answer
- Same-language voice answer
- Ollama local LLM
- Chroma multilingual RAG
- Approved-document sources
- SQLite recent query history
- Clickable recent queries
- Minimal example questions
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import json
import os
import re
import shutil
from urllib.parse import urlsplit

import requests
from supabase import create_client


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.rag_chain import (
    answer_question,
    reset_runtime_cache,
)

from src.ingest import (
    build_vectorstore,
    chunk_documents,
    load_all_documents,
)



def is_official_url(url):
    parsed = urlsplit(url.strip())
    hostname = (parsed.hostname or "").lower().rstrip(".")
    return (
        parsed.scheme == "https"
        and (
            hostname.endswith((".gov.in", ".nic.in"))
            or hostname in {"gov.in", "nic.in"}
        )
    )


def download_official_pdf(url):
    url = url.strip()
    if not is_official_url(url):
        raise ValueError("Use an HTTPS URL from a .gov.in or .nic.in domain.")

    response = requests.get(url, timeout=45, allow_redirects=True)
    response.raise_for_status()
    final_url = response.url
    if not is_official_url(final_url):
        raise ValueError("The download redirected outside an official government domain.")

    content = response.content
    content_type = response.headers.get("content-type", "").lower()
    if not content.startswith(b"%PDF") and "application/pdf" not in content_type:
        raise ValueError("The official URL did not return a PDF document.")

    filename = Path(urlsplit(final_url).path).name or "government-document.pdf"
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return filename, content, final_url


OFFICIAL_DOCUMENTS = {
    "PM-KISAN Operational Guidelines": (
        "https://pmkisan.gov.in/Documents/"
        "RevisedPM-KISANOperationalGuidelines(English).pdf"
    ),
}

from src.config import (
    DATA_DIR,
    SOURCE_DATA_DIR,
    SOURCE_VECTORSTORE_DIR,
    SOURCE_REGISTRY_PATH,
    VECTORSTORE_DIR,
)



SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "scheme-documents")


def supabase_is_configured():
    return bool(
        os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )


def get_supabase_client():
    if not supabase_is_configured():
        raise RuntimeError("Supabase credentials are not configured.")
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


def save_supabase_document(filename, content, source_url):
    client = get_supabase_client()
    client.storage.from_(SUPABASE_BUCKET).upload(
        filename,
        content,
        {"content-type": "application/pdf", "upsert": "true"},
    )
    client.table("documents").upsert(
        {"filename": filename, "source_url": source_url},
        on_conflict="filename",
    ).execute()


def sync_documents_to_local():
    if not supabase_is_configured():
        return False

    client = get_supabase_client()
    rows = client.table("documents").select("filename,source_url").execute().data or []
    if not rows:
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    registry = {}
    changed = False
    for row in rows:
        filename = Path(row["filename"]).name
        content = client.storage.from_(SUPABASE_BUCKET).download(filename)
        target = DATA_DIR / filename
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
            changed = True
        registry[filename] = row.get("source_url", "")

    SOURCE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    return changed

from src.history import (
    add_query,
    get_recent_queries,
)

from src.i18n import (
    SUPPORTED_LANGUAGES,
    text_to_speech,
    speech_to_text_auto,
    detect_language,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=(
        "Smart FAQ Bot — Government Schemes"
    ),

    page_icon="🏛️",

    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --gov-navy: #0B3D6B;
        --gov-navy-dark: #072A4D;
        --gov-gold: #C99A2E;
        --gov-bg: #F5F7FB;
        --gov-card: #FFFFFF;
        --gov-border: #E2E6EE;
        --gov-text: #1F2430;
        --gov-muted: #6B7280;
        --gov-verified: #1E7A46;
        --gov-verified-bg: #EAF6EF;
    }
        --gov-navy-dark: #072A4D;
        --gov-gold: #C99A2E;
        --gov-bg: #F5F7FB;
        --gov-card: #FFFFFF;
        --gov-border: #E2E6EE;
        --gov-text: #1F2430;
        --gov-muted: #6B7280;
        --gov-verified: #1E7A46;
        --gov-verified-bg: #EAF6EF;
    }

    /* ========================================================
       GENERAL
       ======================================================== */

    .stApp {
        background-color: var(--gov-bg);
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 780px;
    }

    /* ========================================================
       HEADER BANNER
       ======================================================== */

    .gov-banner {
        background: linear-gradient(135deg, var(--gov-navy) 0%, var(--gov-navy-dark) 100%);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(11,61,107,0.18);
        position: relative;
        overflow: hidden;
    }

    .gov-banner::after {
        content: "";
        position: absolute;
        top: -20px;
        right: -20px;
        width: 140px;
        height: 140px;
        border: 3px solid rgba(255,255,255,0.08);
        border-radius: 50%;
    }

    .gov-banner-top {
        display: flex;
        align-items: center;
        gap: 14px;
        position: relative;
        z-index: 1;
    }

    .gov-emblem {
        width: 48px;
        height: 48px;
        flex-shrink: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.12);
        border: 2px solid var(--gov-gold);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .gov-banner h1 {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }

    .gov-banner-tag {
        color: var(--gov-gold);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-top: 3px;
    }

    .gov-banner p {
        color: #D7E3F2;
        font-size: 14px;
        margin: 10px 0 0 0;
        line-height: 1.5;
        position: relative;
        z-index: 1;
    }

    /* ========================================================
       DISCLAIMER / TRUST CARD
       ======================================================== */

    .gov-disclaimer {
        background: #FFF8E8;
        border: 1px solid #F0DBA0;
        border-left: 4px solid var(--gov-gold);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 13.5px;
        color: #6B5416;
    }

    .gov-lang-note {
        font-size: 13px;
        color: var(--gov-muted);
        margin-bottom: 20px;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--gov-border);
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--gov-navy);
    }

    section[data-testid="stSidebar"] hr {
        margin-top: 14px;
        margin-bottom: 14px;
        border-color: var(--gov-border);
    }

    /* Scheme chips */
    .scheme-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 6px;
        margin-bottom: 4px;
    }

    .scheme-chip {
        background: #EEF3FA;
        color: var(--gov-navy);
        border: 1px solid #D6E2F2;
        border-radius: 999px;
        padding: 5px 12px;
        font-size: 12.5px;
        font-weight: 500;
        white-space: nowrap;
    }

    .history-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--gov-navy);
        margin-top: 5px;
        margin-bottom: 6px;
    }

    .history-subtitle {
        font-size: 12px;
        color: var(--gov-muted);
        margin-bottom: 10px;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 10px;
        text-align: left;
        border: 1px solid var(--gov-border) !important;
    }

    section[data-testid="stSidebar"] button:hover {
        border-color: var(--gov-navy) !important;
        color: var(--gov-navy) !important;
    }

    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
        border: 1px solid var(--gov-border);
    }

    div[data-testid="stChatMessage"]:nth-of-type(even) {
        background: #EEF3FA;
    }

    div[data-testid="stChatMessage"]:nth-of-type(odd) {
        background: var(--gov-card);
    }

    /* ========================================================
       BADGES: language + verified source
       ======================================================== */

    .gov-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
        margin-top: 4px;
        margin-right: 6px;
    }

    .gov-badge-lang {
        background: #EEF3FA;
        color: var(--gov-navy);
        border: 1px solid #D6E2F2;
    }

    .gov-badge-verified {
        background: var(--gov-verified-bg);
        color: var(--gov-verified);
        border: 1px solid #BFE3CC;
    }

    /* ========================================================
       CHAT INPUT
       ======================================================== */

    div[data-testid="stChatInput"] {
        border-radius: 18px;
        border: 1px solid var(--gov-border);
    }

    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .gov-banner h1 {
            font-size: 21px;
        }
        .gov-emblem {
            width: 38px;
            height: 38px;
            font-size: 19px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pending_question" not in st.session_state:

    st.session_state.pending_question = None


DATA_DIR.mkdir(parents=True, exist_ok=True)
if SOURCE_DATA_DIR.exists() and not any(DATA_DIR.iterdir()):
    for source_file in SOURCE_DATA_DIR.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, DATA_DIR / source_file.name)

VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
if SOURCE_VECTORSTORE_DIR.exists() and not VECTORSTORE_DIR.exists():
    shutil.copytree(SOURCE_VECTORSTORE_DIR, VECTORSTORE_DIR)


if "supabase_sync_done" not in st.session_state:
    st.session_state.supabase_sync_done = True
    if supabase_is_configured():
        try:
            with st.spinner("Syncing saved documents..."):
                remote_documents_changed = sync_documents_to_local()
                if remote_documents_changed or not VECTORSTORE_DIR.exists():
                    build_vectorstore(chunk_documents(load_all_documents()))
                    reset_runtime_cache()
        except Exception as error:
            st.warning(f"Persistent document storage is unavailable: {error}")


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="gov-banner">
        <div class="gov-banner-top">
            <div class="gov-emblem">🏛️</div>
            <div>
                <h1>Smart FAQ Bot</h1>
                <div class="gov-banner-tag">Government of India · Citizen Services Portal</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "Ask about eligibility, documents, benefits, "
    "or how to apply for government schemes."
)


st.caption(
    "⚠️ Answers are generated only from approved "
    "scheme documents. Always verify with the "
    "official source before applying."
)


st.caption(
    "🌐 Type or speak in English, Hindi, Kannada, "
    "Tamil, Telugu, or Malayalam."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # ========================================================
    # ADD OFFICIAL SCHEME DOCUMENTS
    # ========================================================

    st.header("📥 Official government documents")

    st.caption(
        "Documents are downloaded directly from verified government portals. "
        "No manual upload is required."
    )

    selected_document = st.selectbox(
        "Select an approved government guideline",
        list(OFFICIAL_DOCUMENTS),
        key="official_document_choice",
    )

    if st.button(
        "Fetch and index official guideline",
        use_container_width=True,
    ):
        try:
            official_url = OFFICIAL_DOCUMENTS[selected_document]
            filename, content, final_url = download_official_pdf(official_url)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / filename).write_bytes(content)

            source_registry = {}
            if SOURCE_REGISTRY_PATH.exists():
                source_registry = json.loads(
                    SOURCE_REGISTRY_PATH.read_text(encoding="utf-8")
                )
            source_registry[filename] = final_url
            SOURCE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SOURCE_REGISTRY_PATH.write_text(
                json.dumps(source_registry, indent=2),
                encoding="utf-8",
            )

            with st.spinner("Indexing the official document..."):
                build_vectorstore(chunk_documents(load_all_documents()))
                reset_runtime_cache()

            if supabase_is_configured():
                try:
                    save_supabase_document(filename, content, final_url)
                    st.success(
                        f"Added {filename} from the official portal and saved it permanently."
                    )
                except Exception as storage_error:
                    st.warning(
                        "The official document was indexed, but Supabase storage failed. "
                        f"Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY: {storage_error}"
                    )
            else:
                st.success(f"Added {filename} from the official portal.")
        except Exception as error:
            st.error(
                "The official document could not be downloaded or indexed. "
                f"Please try again: {error}"
            )

    st.caption(
        "The source URL is stored with the index. Verify the linked government "
        "portal before relying on an answer."
    )

    st.divider()

    # ========================================================
    # SCHEMES COVERED
    # ========================================================

    st.header(
        "📄 Schemes Covered"
    )


    scheme_files = sorted(
        DATA_DIR.glob("*")
    )


    if scheme_files:

        for file_path in scheme_files:

            if file_path.suffix.lower() in (
                ".pdf",
                ".txt",
            ):

                scheme_name = (
                    file_path.stem
                    .replace("_", " ")
                    .replace("-", " ")
                    .title()
                )


                st.write(
                    f"• {scheme_name}"
                )


    else:

        st.write(
            "No scheme documents found "
            "in data/schemes/"
        )


    # ========================================================
    # TRY ASKING
    # ========================================================

    st.divider()


    st.subheader(
        "💡 Try asking"
    )


    # Keep this section intentionally small.
    example_questions = [

        "Documents needed for Ayushman Bharat?",

        "How to apply for PM-Kisan?",

    ]


    for i, example in enumerate(
        example_questions
    ):

        if st.button(

            example,

            key=f"example_question_{i}",

            use_container_width=True,

        ):

            st.session_state.pending_question = (
                example
            )

            st.rerun()


    # ========================================================
    # RECENT HISTORY
    # ========================================================

    st.divider()


    st.markdown(
        '<div class="history-title">'
        '🕘 Recent Queries'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="history-subtitle">'
        'Click a previous question to ask it again.'
        '</div>',
        unsafe_allow_html=True,
    )


    history = get_recent_queries(
        limit=5
    )


    if history:

        for item in history:

            query = item[
                "question"
            ]

            timestamp = item[
                "timestamp"
            ]


            # ------------------------------------------------
            # TIME
            # ------------------------------------------------

            try:

                dt = datetime.strptime(

                    timestamp,

                    "%Y-%m-%d %H:%M:%S",

                )


                display_time = (
                    dt.strftime(
                        "%I:%M %p"
                    )
                )


            except Exception:

                display_time = timestamp


            # ------------------------------------------------
            # SHORT QUESTION
            # ------------------------------------------------

            display_query = query


            if len(display_query) > 42:

                display_query = (
                    display_query[:39]
                    + "..."
                )


            # ------------------------------------------------
            # CLICKABLE HISTORY
            # ------------------------------------------------

            button_text = (
                f"💬 {display_query}\n"
                f"🕐 {display_time}"
            )


            if st.button(

                button_text,

                key=(
                    f"history_"
                    f"{item['id']}"
                ),

                use_container_width=True,

            ):

                st.session_state.pending_question = (
                    query
                )

                st.rerun()


    else:

        st.caption(
            "No recent queries yet."
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.divider()


    if st.button(

        "🗑️ Clear chat",

        key="clear_chat",

        use_container_width=True,

    ):

        # Only clear current visible chat.
        #
        # History remains in SQLite.

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )


        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        if (
            msg["role"] == "assistant"
            and msg.get("audio")
        ):

            st.audio(

                msg["audio"],

                format="audio/mp3",

            )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if (
            msg["role"] == "assistant"
            and msg.get("sources")
        ):

            st.caption(

                "📄 Source: "
                + ", ".join(
                    msg["sources"]
                )

            )


# ============================================================
# CHAT INPUT
# ============================================================

chat_input = st.chat_input(

    "Ask about a government scheme...",

    accept_audio=True,

    audio_sample_rate=16000,
)


# ============================================================
# QUESTION VARIABLES
# ============================================================

question = None

input_was_voice = False

detected_lang = "en"


# ============================================================
# CHAT INPUT
# ============================================================

if chat_input is not None:

    # ========================================================
    # TEXT
    # ========================================================

    typed_text = ""


    if hasattr(
        chat_input,
        "text",
    ):

        typed_text = (
            chat_input.text
        )


    # ========================================================
    # AUDIO
    # ========================================================

    audio_file = None


    if hasattr(
        chat_input,
        "audio",
    ):

        audio_file = (
            chat_input.audio
        )


    # ========================================================
    # TEXT QUESTION
    # ========================================================

    if (
        typed_text
        and typed_text.strip()
    ):

        question = (
            typed_text.strip()
        )

        input_was_voice = False


    # ========================================================
    # VOICE QUESTION
    # ========================================================

    elif audio_file is not None:

        with st.spinner(
            "🎙️ Listening..."
        ):

            try:

                audio_bytes = (
                    audio_file.read()
                )


                (
                    transcribed,
                    stt_lang,
                ) = speech_to_text_auto(
                    audio_bytes
                )


            except Exception:

                transcribed = None

                stt_lang = None


        if transcribed:

            question = transcribed

            detected_lang = (
                stt_lang or "en"
            )

            input_was_voice = True


        else:

            st.warning(
                "Couldn't understand the audio "
                "clearly. Please try again or "
                "type your question."
            )


# ============================================================
# SIDEBAR QUESTION
# ============================================================

if (
    question is None
    and st.session_state.pending_question
):

    question = (
        st.session_state.pending_question
    )


    st.session_state.pending_question = (
        None
    )


    input_was_voice = False


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # ========================================================
    # DETECT LANGUAGE
    # ========================================================

    if not input_was_voice:

        detected_lang = detect_language(
            question
        )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    add_query(
        question
    )


    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",

            "content": question,
        }

    )


    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )


    # ========================================================
    # RAG
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Checking approved documents..."
        ):

            try:

                (
                    answer,
                    sources,
                    chunks,
                    is_from_web,

                ) = answer_question(

                    question,

                    question_language=(
                        detected_lang
                    ),

                )


            except FileNotFoundError as e:

                answer = (
                    f"⚠️ Setup issue: {e}"
                )

                sources = []

                chunks = []

                is_from_web = False


            except Exception as e:

                answer = (
                    f"⚠️ Something went wrong: {e}"
                )

                sources = []

                chunks = []

                is_from_web = False


        # ====================================================
        # ANSWER
        # ====================================================

        st.write(
            answer
        )


        # ====================================================
        # LANGUAGE
        # ====================================================

        st.markdown(
          f'<span class="gov-badge gov-badge-lang">🌐 '
          f'{SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}</span>',
    unsafe_allow_html=True,
)


        # ====================================================
        # TEXT TO SPEECH
        # ====================================================

        audio_bytes = None


        if input_was_voice:

            audio_bytes = text_to_speech(

                answer,

                detected_lang,

            )


            if audio_bytes:

                st.audio(

                    audio_bytes,

                    format="audio/mp3",

                )


        # ====================================================
        # SOURCE
        # ====================================================

        if is_from_web:

            st.warning(
                "🌐 This answer came from a "
                "web source, not an approved "
                "document."
            )


        elif sources:

            has_official_source = any(
                "https://" in source
                for source in sources
            )

            source_label = (
                "✅ Official source recorded"
                if has_official_source
                else "⚠️ Source URL not recorded"
            )

            st.markdown(
               f'<span class="gov-badge gov-badge-verified">{source_label}: '
               f'{", ".join(sources)}</span>',
        unsafe_allow_html=True,
    )


            # =================================================
            # EXACT TEXT
            # =================================================

            with st.expander(
                "🔍 See the exact text used to answer this"
            ):

                for i, chunk in enumerate(
                    chunks,
                    1,
                ):

                    scheme = (
                        chunk.metadata.get(
                            "scheme",
                            "Unknown",
                        )
                    )


                    document_language = (
                        chunk.metadata.get(
                            "language",
                            "unknown",
                        )
                    )


                    st.markdown(
                        f"**Chunk {i} — "
                        f"{scheme}**"
                    )


                    st.caption(
                        "Document language: "
                        + document_language
                    )


                    preview = (
                        chunk.page_content[:400]
                    )


                    if len(
                        chunk.page_content
                    ) > 400:

                        preview += "..."


                    st.text(
                        preview
                    )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "assistant",

            "content": answer,

            "audio": audio_bytes,

            "sources": sources,

            "is_from_web": is_from_web,

        }

    )