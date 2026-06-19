# Secure Contextual RAG Pipeline

A pure-Python RAG pipeline built directly on the Anthropic API — no LangChain, no abstractions. The goal is to understand every stage of the pipeline and empirically compare retrieval strategies, including using Claude itself as the chunking engine, and caching as an alternative to chunking entirely.

---

## Three Approaches Compared

**Method 1 — No RAG.** Baseline: Claude answers from training knowledge alone, no document access.

**Method 2 — RAG with hybrid agentic + structural chunking.**
1. Claude reads the full document and returns verbatim phrases marking topic boundaries
2. The document is cut at those phrases into semantic chunks
3. Any chunk still over a size threshold is split further — by paragraph, then by sentence, then (last resort) a hard character cut — so nothing oversized survives to embedding
4. Chunks are embedded (`sentence-transformers`, `all-MiniLM-L6-v2`, 384-dim) and indexed in FAISS (`IndexFlatL2`)
5. Per query: question is embedded, top-k chunks retrieved, sent to Claude as context

**Method 3 — CAG (cached full document, no chunking).** The entire document is sent to Claude on every call, marked as cacheable. No chunking, no embedding, no retrieval — Claude reads the whole document each time. Cheap on repeat questions against the same document (cache reuse); more expensive on a single one-off question (cache-write premium). Only viable while the document fits in the model's context window.

---

## Security Design

Two independent layers, applied to both the user's question and any retrieved document content:

**Layer 1 — Blocklist filter.** `is_safe_input()` scans text for known prompt-injection phrases before anything reaches Claude.

**Layer 2 — Structural isolation.** Document content is wrapped in `<untrusted_documents>` XML tags, signaling to Claude at a structural level that it's data, not instructions — independent of the blocklist.

---

## Experiment 01 — Does RAG Actually Improve Answers?

**Question:** *"What specific claim does Bauerlein make about digital technology?"*
**Corpus:** 2 PDFs — `1_Four_Frameshifts_Ch1.pdf` + `3_Humanizing_Pedagogy_Ch7.pdf` (23,947 chars total)

| Method | Tokens | Time | Result |
|:---|:---|:---|:---|
| 1: No RAG | ~200 | ~2.5s | Doesn't know — no document access |
| 2: Chunking (unbounded, pre-fix) | 5,616 | 3.06s | Correct |
| 2: Chunking (hybrid safety valve) | 2,498 | 2.8–3.4s | **Wrong** — "I don't know" |
| 3: CAG (cached full doc) | ~6,010* | 1.1s | Correct |

\* includes cache-write tokens, billed at a different rate than normal input tokens.

**Key finding:** fixing chunking's oversized-chunk problem (via a paragraph/sentence/hard-cut safety valve) introduced a new failure mode — splitting a chunk can separate a fact from the surrounding context that made it retrievable, causing a correct answer to be missed entirely even though it's in the corpus. CAG sidesteps this because there are no chunk boundaries for a fact to be split across. Full writeup of all three tradeoffs (unbounded chunking, sentence-fallback fragmentation, and CAG's own limits) in each `experiment.txt`.

---

## Tech Stack

| Component | Library |
|:---|:---|
| LLM | `anthropic` — `claude-haiku-4-5-20251001` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector store | `faiss-cpu` — `IndexFlatL2` |
| PDF parsing | `pdfplumber` |


---

 Note -- This repo is an active empirical lab for studying the tradeoffs of framework-less RAG pipelines, not a finished product. Findings and failures are tracked in each experiment.txt as they come up.