# 🏛️ Smart FAQ Bot for Government Scheme Queries

**CODE CIRCUIT 2026 — National Level Hackathon**
Problem Statement: SW-01-L — Smart FAQ Bot for Government Scheme Queries

## The Problem

Citizens often can't benefit from government schemes they're eligible for —
not because the schemes don't exist, but because eligibility rules, required
documents, benefits, and application steps are buried in long, jargon-heavy
official PDFs scattered across different websites. Every year, thousands of
crores in benefits go unclaimed simply because people don't know a scheme
exists, don't understand if they qualify, or don't know how to apply.

## The Solution

A **document-grounded chatbot** that answers citizens' questions using ONLY
approved government scheme documents — never guessing, never hallucinating.
Every answer is shown alongside the exact scheme and source document it came
from, so users can verify it and go straight to the right official channel.

## How It Works (RAG — Retrieval-Augmented Generation)

```
Government scheme documents (PDF/TXT)
            │
      [Split into chunks]
            │
      [Convert to embeddings] ──► [Chroma Vector Database]
                                          │
User question ──────► [Retrieve top matching chunks] ◄──┘
            │
      [LLM answers using ONLY retrieved chunks]
            │
      Simple-language answer + source citation
```

1. **Ingestion** — Official scheme PDFs are split into small chunks and stored
   in a vector database (Chroma), tagged with which scheme they belong to.
2. **Retrieval** — When a user asks a question, we search the vector database
   for the chunks most similar *in meaning* to the question (not just keyword
   matching).
3. **Generation** — Those chunks are given to an LLM with a strict instruction:
   *"Answer only using this context. If it's not here, say you don't know."*
   This is what makes the bot trustworthy instead of a generic chatbot.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| RAG orchestration | LangChain |
| Vector Database | Chroma |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| LLM | Claude (Anthropic API) |
| UI | Streamlit |

## Project Structure

```
smart-faq-bot/
├── data/schemes/          # Government scheme documents (PDF or TXT)
├── vectorstore/           # Auto-generated vector database (created by ingest.py)
├── src/
│   ├── config.py          # All settings — paths, model names, prompt template
│   ├── ingest.py          # Builds the vector database from documents
│   └── rag_chain.py       # Retrieval + LLM answer logic
├── app.py                 # Streamlit chat interface
├── requirements.txt       # Python dependencies
└── .env.example           # Template for your API key
```

## Setup Instructions

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and paste your Anthropic API key (get one at
https://console.anthropic.com):

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

### 3. Add your scheme documents

Sample placeholder documents for **PM-Kisan, Ayushman Bharat, PM Awas Yojana,
and National Scholarship** are already included in `data/schemes/` so the
project runs out of the box.

**Before your final demo, replace these with the real official PDFs** downloaded
from `.gov.in` sources (e.g., pmkisan.gov.in, pmjay.gov.in, pmaymis.gov.in,
scholarships.gov.in) — just drop the PDFs into `data/schemes/`, using the
same filenames or new ones. The code reads both `.pdf` and `.txt` files
automatically.

### 4. Build the vector database

Run this once (and again any time you add/change documents):

```bash
python src/ingest.py
```

You should see output confirming documents were loaded, chunked, and saved.

### 5. (Optional) Quick terminal test

Before launching the UI, you can sanity-check the bot in the terminal:

```bash
python src/rag_chain.py
```

Type a question like `Am I eligible for PM-Kisan if I own 3 acres of land?`
and confirm it gives a sensible, sourced answer.

### 6. Launch the app

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

This opens a browser window with the chat interface.

On Windows, when the project was copied from another computer, use the installed
Python interpreter instead of the old virtual environment:

```powershell
py -3.13 -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### 7. Public deployment

Run ingestion once, commit the generated `vectorstore/` folder, then push the
project to GitHub and create a Streamlit Community Cloud app from the
repository. Set the main file to `app.py`.

```bash
python -m src.ingest
```

For a container host such as Render, add a Docker service that runs
`pip install -r requirements.txt`, then `python -m src.ingest`, and starts:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Before deploying, set these Render environment variables:

```text
OLLAMA_BASE_URL=https://your-reachable-ollama-server.example.com
OLLAMA_MODEL=llama3.2:3b
```

`OLLAMA_BASE_URL` must point to an internet-reachable Ollama server. The default
`http://localhost:11434` only works when Ollama is running on the same machine
as the app, so it cannot be used by a public cloud deployment.

### 8. Persistent document storage with Supabase

Create a free Supabase project, then create a private Storage bucket named
`scheme-documents`. In the Supabase SQL editor, create the metadata table:

```sql
create table documents (
  filename text primary key,
  source_url text not null,
  created_at timestamptz not null default now()
);
```

Add these values to Streamlit Cloud under **App settings > Secrets**:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
SUPABASE_BUCKET = "scheme-documents"
```

Use the service-role key only in Streamlit Cloud Secrets. Never commit it to
GitHub or put it in a public `.env` file. The app stores only PDFs accepted from
official `.gov.in` or `.nic.in` URLs and restores them from Supabase after a
restart.

## Example Questions to Demo

- "Am I eligible for PM-Kisan if I own 3 acres of land?"
- "What documents do I need for Ayushman Bharat?"
- "How much financial help do I get under PM Awas Yojana?"
- "How do I apply for a national scholarship?"
- Try something **out of scope** (e.g., "Can I apply for this from Canada?")
  to demonstrate the bot correctly says it doesn't have that information,
  instead of guessing.

## Why This Approach (Talking Points for Judges)

- **Grounded, not generic**: The bot can only answer from approved documents —
  it's built to say "I don't know" rather than hallucinate, which matters when
  the answer affects a real government benefit decision.
- **Traceable**: Every answer cites the scheme and source document, satisfying
  the requirement to "point the user to the source document."
- **Scalable**: Adding a new scheme is as simple as dropping a new PDF into
  `data/schemes/` and re-running `ingest.py` — no code changes needed.
- **Low-cost & accessible**: Uses a free local embedding model; only the final
  answer generation calls a paid API, keeping the system cheap to run at scale.

## Possible Next Steps (mention if asked about future scope)

- Regional language support (Hindi, Kannada, etc.) for wider accessibility
- Voice input for users with low digital/text literacy
- WhatsApp integration for reach beyond web/app users
- Expanding to hundreds of schemes via automated document ingestion pipelines

## Team

*(Add your team name and member names here before submission.)*
