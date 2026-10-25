# Knowledge Hub: Document Q&A with RAG

Upload company documents (PDF, Word, Markdown, text) and ask questions in plain English. Knowledge Hub finds the most relevant passages and has an AI model answer **only from those passages**, citing each source as `[S1]`, `[S2]`, … with a confidence score.

Built with **FastAPI**, **SQLite** and a single-file HTML frontend. Works with a free local model (Ollama), the OpenAI API, or any OpenAI-compatible service, and still works with no AI at all.

## How it works

```
upload ─► extract text (pypdf / python-docx) ─► split into ~1,200-char chunks ─► embed ─► SQLite
question ─► embed ─► cosine similarity over chunks ─► top 4 passages ─► chat model ─► answer + [S1] citations
```

- **Chunking** keeps paragraphs together, splits long PDF text at sentence boundaries, and overlaps neighbouring chunks so facts aren't cut in half.
- **Embeddings are tagged with the model that made them.** Switching models re-embeds automatically instead of silently comparing incompatible vectors. Vectors are cached in memory after first use.
- **Graceful fallback.** If the AI service is down or not configured, retrieval switches to BM25 keyword search and the app returns the best matching passages, clearly labelled, instead of failing.
- **Guardrails.** Questions containing passwords or API keys are refused; the model is told not to invent facts outside the sources.
- **Public-demo mode** (`DEMO_MODE=true`): per-IP rate limits, smaller uploads, a cap on stored documents, sample documents protected from deletion, and each visitor sees only their own question history.

## Run locally (free, with Ollama)

1. Install [Ollama](https://ollama.com/download), then download the two models:
   ```powershell
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
2. Set up Python and start the server:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   python -m uvicorn backend.main:app --reload
   ```
3. Open http://127.0.0.1:8000/

The first answer can take a while as Ollama loads the model. On a computer with little memory, set `OLLAMA_CHAT_MODEL=qwen2.5:3b` in `.env` and run `ollama pull qwen2.5:3b`.

**Other providers:** set `AI_PROVIDER=openai` and `AI_API_KEY=...` for OpenAI, or `AI_PROVIDER=compatible` with `AI_BASE_URL`, `AI_API_KEY`, `AI_CHAT_MODEL` and `AI_EMBEDDING_MODEL` for any other OpenAI-compatible service. With `AI_PROVIDER=none` the app runs in keyword-search mode.

## Deploy a public demo

`render.yaml` deploys the app to [Render](https://render.com) on the free plan: **New + → Blueprint →** choose this repository. It starts in demo mode with keyword search; add `AI_PROVIDER` and `AI_API_KEY` in the Render dashboard to turn AI answers on. A `Dockerfile` is included for other hosts.

Free hosting has no permanent disk, so uploads reset when the service restarts; the sample documents are reloaded automatically.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Provider, models, mode (`ai` or `retrieval_only`) |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/documents` | Upload and index a file (multipart `file`) |
| DELETE | `/api/documents/{id}` | Remove a document and its chunks |
| POST | `/api/chat` | `{question, history}` → answer, sources, confidence |
| GET | `/api/history` | Recent questions (per visitor via `X-Client-Id`) |
| GET | `/api/stats` | Document, chunk and question counts; retrieval hit rate |

Interactive docs: http://127.0.0.1:8000/docs

## Tests

```powershell
python -m unittest discover -s tests -v
```

24 tests cover parsing (PDF, DOCX with tables, Markdown), chunking, keyword ranking, semantic search with a fake AI provider, model switching, fallback when the AI is down, demo-mode limits, per-visitor history and the analytics numbers.

## Project layout

```
backend/
  main.py     API routes, retrieval, answer generation, rate limiting
  ai.py       OpenAI-compatible client (embeddings, chat, readiness check)
  store.py    SQLite schema, migrations, vector cache, stats
  text.py     file parsing, chunking, BM25 keyword scoring
  config.py   settings from environment variables
  seeds.py    sample documents for a fresh install
tests/        unittest suite
ai-knowledge-base.html   frontend
```
