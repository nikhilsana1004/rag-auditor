# rag-auditor 🔍

**CLI tool to audit any RAG pipeline for failure modes** — before they reach production.

Point it at your documents, give it a query, get a full diagnostic report.

```
┌─────────────────────────────────────────────────────────────┐
│  RAG Audit Summary                                          │
│  ┌──────────────────────────────┬────────┬───────┬───────┐  │
│  │ Check                        │ Status │ Score │ Issue │  │
│  ├──────────────────────────────┼────────┼───────┼───────┤  │
│  │ #1 Bad Chunking              │ ✓ PASS │  84%  │  ...  │  │
│  │ #2 Poor Retrieval            │ ⚠ WARN │  61%  │  ...  │  │
│  │ #3 Lost in the Middle        │ ✗ FAIL │  38%  │  ...  │  │
│  │ #4 Retriever-Generator Mismatch│ ⚠ WARN│ 54%  │  ...  │  │
│  └──────────────────────────────┴────────┴───────┴───────┘  │
│  Overall pipeline health: 59% — ⚠ Needs attention           │
└─────────────────────────────────────────────────────────────┘
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

No API keys required. Embeddings run locally via `sentence-transformers` (all-MiniLM-L6-v2).

---

## Quick start

```bash
# Audit a folder of docs against a query
rag-audit run --docs ./my_docs --query "What is the refund policy?"

# Try the bundled sample
rag-audit run --docs docs/sample --query "How long does a refund take?"

# Run specific checks only
rag-audit run --docs ./docs --query "..." --checks chunking,retrieval

# Tune chunk size and top-k
rag-audit run --docs ./docs --query "..." --chunk-size 256 --top-k 3

# Verbose output — see chunk-level detail
rag-audit run --docs ./docs --query "..." --verbose

# Save JSON report
rag-audit run --docs ./docs --query "..." --output report.json

# List all available checks
rag-audit list-checks
```

---

## Supported document formats

| Format | Library used |
|--------|-------------|
| `.txt` / `.md` | built-in |
| `.pdf` | `pdfplumber` (fallback: `pypdf`) |
| `.docx` | `python-docx` |

---

## Project structure

```
rag-auditor/
├── rag_auditor/
│   ├── cli.py                    # Typer CLI entry point
│   ├── ingest/
│   │   └── loader.py             # Document loader (PDF, DOCX, TXT, MD)
│   ├── auditors/
│   │   ├── chunking.py           # #1 Bad chunking detector
│   │   ├── retrieval.py          # #2 Retrieval recall auditor
│   │   ├── lost_in_middle.py     # #3 Position bias auditor
│   │   └── rg_alignment.py       # #4 Retriever-generator mismatch
│   └── utils/
│       ├── chunker.py            # Fixed + sentence chunkers
│       ├── embeddings.py         # Local embeddings (sentence-transformers)
│       └── report.py             # Rich terminal report + JSON export
├── docs/sample/
│   └── acme_refund_policy.txt    # Sample doc for quick testing
├── tests/
├── pyproject.toml
└── README.md
```

---

## Output example

```
╭─ RAG Auditor — failure mode detector ──────────────────────╮
│ docs: ./docs   query: What is the refund policy?            │
╰─────────────────────────────────────────────────────────────╯

#1 Auditing: Bad Chunking
  Status: WARN  score: 64%
  ⚠  18/42 chunks exceed 600 tokens. Large chunks dilute relevant content.
  →  Reduce chunk_size or switch to sentence chunking.

#2 Auditing: Poor Retrieval / Low Recall
  Status: PASS  score: 78%
  Top-1 similarity: 0.73 — retrieval looks healthy.

#3 Auditing: Lost in the Middle
  Status: FAIL  score: 35%
  ⚠  Best chunk is at position #3 of 5. Attention weight: 35%.
  →  Add a cross-encoder reranker to promote the best chunk to position #1.

#4 Auditing: Retriever-Generator Mismatch
  Status: WARN  score: 55%
  ⚠  Moderate mismatch: 40% of top-5 chunks differ.
  →  Use HyDE: generate a hypothetical answer, embed that, then retrieve.
```

---

## Research references

- **Lost in the Middle**: Liu et al., 2023 — position bias in LLM context windows
- **RAG-E**: Jan 2026 — retriever-generator misalignment in 47–67% of real queries
- **HyDE**: Gao et al., 2022 — hypothetical document embeddings for better retrieval

---

## License

MIT
