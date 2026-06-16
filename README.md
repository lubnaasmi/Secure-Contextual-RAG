## Secure Contextual RAG Pipeline

A secure RAG pipeline built in pure Python using the Anthropic API directly — no LangChain, no abstractions. The goal is to understand and measure every stage of the pipeline, and to experiment with using Claude itself as the chunking engine.


### What This Does

The pipeline has two phases:

 
 **Ingestion (runs once)**


1. Load PDFs from a folder using pdfplumber
2. Send the full document text to claude-haiku-4-5-20251001 to identify where major topic boundaries are — returning verbatim 15-20 word phrases marking each split point
3. Cut the document at each phrase to produce semantic chunks
4. Embed each chunk into a 384-dimensional vector using sentence-transformers (all-MiniLM-L6-v2)
5. Store all vectors in a FAISS IndexFlatL2 index for similarity search


 **Query (runs per question)**


1. Scan the question for prompt injection patterns (is_safe_input)
2. Embed the question into the same vector space as the chunks
3. FAISS retrieves the top-3 most similar chunks
4. Scan retrieved chunks for injection patterns
5. Send chunks to Claude wrapped in <untrusted_documents> tags with the question



### Security Design

Two independent layers:

**Layer 1** — Blocklist filter: is_safe_input() scans both the user question and retrieved chunks for known injection phrases before anything reaches Claude.

**Layer 2** — Structural isolation: Retrieved document content is wrapped in <untrusted_documents> XML tags. This signals to Claude at a structural level that the content is data, not instructions — independent of the blocklist.

Both layers run on input AND on retrieved chunks, protecting against attacks hidden inside documents.


**Experiment 01 — Does RAG Actually Improve Answers?**

**Question:** "What specific claim does Bauerlein make about digital technology?"

**Corpus:** 2 PDFs — 1_Four_Frameshifts_Ch1.pdf + 3_Humanizing_Pedagogy_Ch7.pdf (23,947 chars total)


**Key Design Decision: Claude as Chunker**

Instead of fixed-size character splitting, Claude reads the document and returns verbatim phrases (15-20 words) marking where topics change. The document is then cut at each phrase.

Why this is interesting: It produces semantically coherent chunks rather than arbitrary cuts. A chunk about "Humanizing Pedagogies" won't be split mid-argument.

**Known tradeoffs:**


- Adds one API call per ingestion run
- Phrase matching via .find() requires Claude to reproduce text exactly — longer phrases (15-20 words) are more reliable than short ones
- Academic PDFs with LaTeX encoding (like arXiv papers) can have spacing issues in extracted text, making phrase matching harder
- Small documents may produce only 1 chunk if Claude finds no major topic changes



### Tech Stack

LLManthropic — claude-haiku-4-5-20251001 <br>
Embeddingssentence-transformers — all-MiniLM-L6-v2 (384-dim) <br>
Vector storefaiss-cpu — IndexFlatL2 <br>
PDF parsingpdfplumber <br>


### Setup

`pip install anthropic pdfplumber sentence-transformers faiss-cpu numpy
export ANTHROPIC_API_KEY=your-key-here`

### Run the pipeline:

`cd pure_python_version
python rag.py`

### Run the experiment:

bashcd pure_python_version/experiments
python exp_01_method_comparison.py | tee results/exp_01_output.txt

Add PDFs to the data/ folder. The pipeline loads all .pdf files from that directory automatically.


> **Note:** PDFs are not included in this repo due to copyright. 
> Add your own PDFs to the `data/` folder to run the pipeline.

**Active Research & Iteration Logs**
I am actively treating this repository as an empirical laboratory to study the trade-offs of framework-less data pipelines