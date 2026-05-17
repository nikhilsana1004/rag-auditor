# rag-auditor 🔍

> CLI tool to audit any RAG pipeline for failure modes — chunking, retrieval, context bias, and retriever-generator mismatch.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Point it at your documents, give it a query, get a full diagnostic report — with optional LLM citation checking and reranker integration.

```
╭─────────────────────────────────────────────────────────────╮
│  RAG Audit Summary                                          │
│  ┌──────────────────────────────┬────────┬───────┐          │
│  │ Check                        │ Status │ Score │          │
│  ├──────────────────────────────┼────────┼───────┤          │
│  │ #1 Bad Chunking              │ ✓ PASS │  84%  │          │
│  │ #2 Poor Retrieval            │ ⚠ WARN │  61%  │          │
│  │ #3 Lost in the Middle        │ ✗ FAIL │  35%  │          │
│  │ #4 Retriever-Generator Mismatch│ ✓ PASS│ 91%  │          │
│  └──────────────────────────────┴────────┴───────┘          │
│  Overall pipeline health: 68% — ⚠ Needs attention           │
╰─────────────────────────────────────────────────────────────╯
```

---

## Failure modes detected

| # | Failure | What it catches |
|---|---------|-----------------|
| 1 | **Bad chunking** | Oversized chunks, undersized chunks, mid-sentence splits |
| 2 | **Poor retrieval** | Low recall@k — the right chunk isn't in your top-k |
| 3 | **Lost in the middle** | Best chunk buried in context window (Liu et al., 2023) |
| 4 | **Retriever-Generator mismatch** | LLM ignores top-ranked chunks (RAG-E, Jan 2026) |

---

## Installation

```bash
git clone https://github.com/nikhilsana1004/rag-auditor
cd rag-auditor
pip install -e .
```

---

## Configuration — API keys via .env

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM providers — fill in whichever you use
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxx

# Reranker
COHERE_API_KEY=xxxxxxxxxx

# Optional: override default models
# RAG_AUDIT_LLM_MODEL=claude-3-5-sonnet-20241022
# RAG_AUDIT_RERANKER_MODEL=rerank-english-v3.0
```

Keys are loaded automatically from `.env` — no need to pass them on the command line. The `.env` file is gitignored and never committed.

---

## Quick start

```bash
# Basic audit — no API key needed, runs fully local
python -m rag_auditor.cli run --docs docs/sample --query "How long does a refund take?"

# With your own documents
python -m rag_auditor.cli run --docs ./my_docs --query "What is the cancellation policy?"
```

---

## LLM integration

Adds a real LLM citation check to auditor #4. Without this flag, auditor #4 uses a HyDE proxy approximation instead.

### Anthropic (Claude)

```bash
# Key is read from ANTHROPIC_API_KEY in .env automatically
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --llm anthropic

# Override the default model (claude-3-5-haiku-20241022)
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --llm anthropic \
  --llm-model claude-3-5-sonnet-20241022
```

### OpenAI (GPT)

```bash
# Key is read from OPENAI_API_KEY in .env automatically
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --llm openai

# Override the default model (gpt-4o-mini)
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --llm openai \
  --llm-model gpt-4o
```

**What the LLM check does:** sends your top-k chunks numbered as `[CHUNK 1]`, `[CHUNK 2]` etc. and asks the LLM to answer the query and cite which chunks it used. If the LLM ignores what the retriever ranked #1, that is flagged as real retriever-generator misalignment.

---

## Reranker integration

Re-scores and re-orders retrieved chunks before they enter the context window. Auditor #3 shows before/after position and attention weight.

### Cross-encoder (local, free)

No API key needed. Downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) on first use.

```bash
pip install sentence-transformers

python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --reranker cross-encoder
```

### Cohere Rerank API

```bash
pip install cohere

# Key is read from COHERE_API_KEY in .env automatically
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --reranker cohere

# Override the default model (rerank-english-v3.0)
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --reranker cohere \
  --reranker-model rerank-multilingual-v3.0
```

**What the reranker does:** takes your top-k chunks from the bi-encoder retriever and re-scores each one by looking at the query and chunk *together* (cross-encoder). The best chunk is promoted to position #1 in the context window — fixing the lost-in-the-middle problem.

---

## Full audit — LLM + reranker combined

```bash
python -m rag_auditor.cli run --docs ./my_docs \
  --query "What is the refund policy?" \
  --llm anthropic \
  --reranker cross-encoder \
  --chunk-size 256 \
  --top-k 5 \
  --output report.json \
  --verbose
```

---

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--docs` | required | Path to folder or single file |
| `--query` | required | Test query to audit against |
| `--chunk-size` | 512 | Tokens per chunk |
| `--chunk-overlap` | 50 | Token overlap between chunks |
| `--top-k` | 5 | Chunks to retrieve |
| `--checks` | all | Comma-separated: `chunking,retrieval,lost_in_middle,rg_alignment` |
| `--llm` | none | LLM provider: `anthropic` or `openai` |
| `--llm-key` | from .env | API key (optional if set in .env) |
| `--llm-model` | provider default | Override model name |
| `--reranker` | none | Reranker: `cross-encoder` or `cohere` |
| `--reranker-key` | from .env | API key for Cohere (optional if set in .env) |
| `--reranker-model` | provider default | Override reranker model |
| `--output` | none | Save JSON report to this path |
| `--verbose` | false | Show chunk-level detail |

---

## Supported document formats

| Format | Library |
|--------|---------|
| `.pdf` | pdfplumber (fallback: pypdf) |
| `.docx` | python-docx |
| `.txt` / `.md` | built-in |

---

## Install only what you need

```bash
pip install -e .                          # base — chunking + retrieval (TF-IDF embeddings)
pip install -e ".[embeddings]"            # + local sentence-transformers + cross-encoder
pip install -e ".[embeddings,anthropic]"  # + Claude
pip install -e ".[embeddings,openai]"     # + GPT
pip install -e ".[embeddings,cohere]"     # + Cohere reranker
pip install -e ".[all]"                   # everything
```

---

## Project structure

```
rag-auditor/
├── rag_auditor/
│   ├── cli.py                     # Typer CLI — all flags and entry point
│   ├── ingest/
│   │   └── loader.py              # PDF, DOCX, TXT, MD ingestion
│   ├── auditors/
│   │   ├── chunking.py            # #1 Bad chunking detector
│   │   ├── retrieval.py           # #2 Retrieval recall auditor
│   │   ├── lost_in_middle.py      # #3 Position bias + reranker fix
│   │   └── rg_alignment.py        # #4 RG mismatch — HyDE proxy or real LLM
│   ├── llm/
│   │   ├── base.py                # BaseLLM interface + citation parser
│   │   ├── anthropic_llm.py       # Claude integration
│   │   └── openai_llm.py          # GPT integration
│   ├── reranker/
│   │   ├── base.py                # BaseReranker interface
│   │   ├── crossencoder.py        # Local MS-MARCO cross-encoder
│   │   └── cohere_reranker.py     # Cohere Rerank API
│   └── utils/
│       ├── chunker.py             # Fixed + sentence chunkers
│       ├── embeddings.py          # Local embeddings + TF-IDF fallback
│       └── report.py              # Rich terminal table + JSON export
├── docs/sample/                   # Sample doc for quick testing
├── tests/                         # 18 pytest tests
├── .env.example                   # API key template
└── pyproject.toml
```

---

## Research references

- **Lost in the Middle** — Liu et al., 2023: position bias in LLM context windows
- **RAG-E** — Jan 2026: retriever-generator misalignment in 47–67% of real queries
- **HyDE** — Gao et al., 2022: hypothetical document embeddings for better retrieval

---

## Roadmap

- [ ] HyDE auto-fix (re-retrieve with hypothetical answer embedding)
- [ ] Hybrid search (BM25 + dense) comparison
- [ ] RAGAS integration for answer quality scoring
- [ ] GraphRAG support for multi-hop queries
- [ ] HTML report output

---

## License

MIT
