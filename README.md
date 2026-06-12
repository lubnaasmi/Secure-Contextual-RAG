
# Secure Contextual RAG Pipeline

A secure, context-aware RAG pipeline built in pure Python using the native Anthropic API. This project is configured as an empirical systems experiment comparing local parametric knowledge base lookups against an agentic-chunked, vector-indexed document retrieval engine.

##  What I Did (Core Implementation)
I bypassed heavy orchestration frameworks (like LangChain) to write a transparent, production-grade retrieval pipeline from scratch in raw Python. 

The pipeline implements:
1. **Direct PDF Ingestion:** Uses `pdfplumber` to extract text streams page-by-page.
2. **Agentic Boundary Detection:** Passes segmented document buffers to `claude-haiku-4-5-20251001` to identify explicit semantic and thematic split phrases.
3. **Local Vector Math:** Transforms raw text chunks into 384-dimensional dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) and a local `faiss-cpu` (IndexFlatL2) Euclidean distance index.
4. **Targeted Security Filtering:** * Pre-scans incoming user strings for common adversarial instructions (`is_safe_input`).
   * Wraps retrieved context strings in strict structural <untrusted_documents> XML tags to isolate raw data blocks from system prompts.
---

## Current Empirical Test Results

### Parameters
* **Query:** *"What is retrieval augmented generation?"*
* **Dataset:** Foundational RAG Research Paper (`RAG_NLP.pdf` — 62,602 characters)

| Methodology | Pipeline Latency | Token Overhead | Retrieval Time | Response Profile & Grounding Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Method 1: No RAG** | ~4.48s | 329 tokens | N/A | **High-Level / Generic:** Relies on Claude's internal parametric weights. Provides a clean but generalized textbook overview. |
| **Method 2: Claude Split** | ~7.35s | 2,530 tokens | 0.038s | **Highly Academic / Grounded:** Captures verbatim non-parametric language directly from the source paper (*"fine-tuning approach that endows pre-trained, parametric-memory..."*). |

---

#### The Red Flag: The `[:12000]` Hard-Slice Constraint

During initial data inspection, a major structural asymmetry was discovered: **Chunk 11 contained 50,683 characters (over 80% of the entire paper)**, while Chunks 1–10 safely averaged ~1,500 characters.

#### The Technical "Why"
This failure mode was directly caused by an intentional safety gate on Line 50 of `rag.py`:

`{full_text[:12000]}` 


 
- **The Constraint:**  To prevent output token overflow and manage initial API token budgets, the ingest pipeline strictly truncated the document buffer passed to the Claude chunking loop at the first 12,000 characters.

- **The Consequence:** Claude beautifully mapped out section splits for the Abstract, Introduction, and early Methodology layers because they sat inside that 12k buffer. However, because the rest of the text was physically cut off, the remaining 50k+ characters of the document fell into a final single, un-tokenized "tail accumulation" block.

- **Downstream Cost:** When the vector index hits this mega-chunk, it forces the generator to ingest a massive payload, driving query token usage up to ~2,500 tokens and neutralizing the targeted extraction benefits of a RAG pipeline.

###  Tech Stack

* **LLM Engine:** Native Anthropic Client (`claude-haiku-4-5-20251001`)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Store:** `faiss-cpu` (Direct Index Flat L2 optimization)
* **PDF Parser:** `pdfplumber`



###  Setup
```bash
pip install anthropic pdfplumber sentence-transformers faiss-cpu numpy
export ANTHROPIC_API_KEY=your-key-here
cd pure_python_version
python rag.py
```