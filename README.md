# Multi-Modal RAG with LangChain (PDF + Images)

**One vector space (CLIP) → unified retrieval over text *and* images → GPT-4.1 answers grounded on the exact page text & figure**

> **TL;DR.** This project builds a **multimodal RAG** system that understands PDFs containing **text and embedded images**. It extracts page text + figures with **PyMuPDF**, embeds both using **CLIP ViT-B/32** (a single, shared embedding space), retrieves the most relevant items for a query (across modalities), formats them in the **multimodal message** that GPT-4.1 expects, and returns a **grounded** answer with the supporting page snippets and images.

---

## Project Overview

Traditional (text-only) RAG misses answers hidden in **charts, plots, and scanned figures**. This repo unifies **text + images** so questions like “*What does the chart on page 1 show?*” retrieve the correct **figure** *and* the nearby **text** in one pass. The solution uses:

* **PyMuPDF (fitz)** to parse PDFs and extract both **text** and **embedded images**
* **CLIP (openai/clip-vit-base-patch32)** to produce **512-d unit-normalized embeddings** for **both** modalities
* **FAISS** vector store built **from precomputed embeddings**
* A **LangChain** pipeline that performs **cross-modal retrieval** and composes a **multimodal prompt** (text + `image_url` with base64 data) for **GPT-4.1**

---

## Objectives

* **Unify** text and image evidence in a **single** vector index (no captioning required).
* Return **grounded** answers that reference the **exact page text** and **figure(s)** used.
* Keep the system **simple and fast**: one embedding model (CLIP), one FAISS store, thin LangChain wrappers.

---
## Why Multimodal RAG?

Many “text” PDFs hide decisive information in graphics: bar charts, screenshots, logos, tables rendered as pixels, etc. A text-only pipeline can’t retrieve these. CLIP (Contrastive Language–Image Pretraining) gives us a shared latent space so:

* Text queries can match images that depict the answer.
* Image-derived captions aren’t required (though you can add them later).
* One retriever searches text + images together.

<img src="https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain/blob/main/clip_embedding_space.png" width="400"/>
---
## Repository / Notebook

* **`multimodalopenai.ipynb`** – the complete implementation. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain/blob/main/multimodalopenai.ipynb)

> The code expects a sample PDF named **`multimodal_sample.pdf`** (contains text + a bar chart).

---

## System Architecture

```
PDF (text + figures)
        │
        ├─ PyMuPDF → page text ─┐
        └─ PyMuPDF → images  ───┴─→ build LangChain Documents (metadata: type, page, image_id)
                │
   CLIP text encoder            CLIP image encoder
                │                         │
        512-d unit vectors ← L2-normalized for both
                └──────────── merge → FAISS.from_embeddings (with doc metadata)
                                     │
User query ──CLIP(text)→ embedding ──┴─ similarity_search_by_vector → top-k (text + image docs)
                                     │
            create **multimodal** ChatMessage (text blocks + base64 images)
                                     │
                            GPT-4.1 (vision) → grounded answer
```
<img src="https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain/blob/main/multimodal_rag_architecture.png" width="700"/>

---

## Implementation Details (code-backed)

### 1) PDF parsing → **LangChain Documents**

* Uses **PyMuPDF** (`fitz`) to open the PDF and iterate pages.
* **Text**: `page.get_text()` → wrap each chunk as a `Document` with metadata `{"page": i, "type": "text"}`; chunk via `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`.
* **Images**: `page.get_images()` → extract pixmaps → convert to **PIL Image** → store a **base64** copy in `image_data_store[image_id]` → create a `Document` with metadata `{"page": i, "type": "image", "image_id": image_id}`.

### 2) **Unified embeddings** with CLIP

* Loaded once:

  ```python
  clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
  clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  ```
* Helpers (from the notebook):

  * **`embed_text(text)`** → `clip_model.get_text_features(...)`

    * tokenizer via `clip_processor`, `max_length=77`, **L2-normalize** to unit vectors.
  * **`embed_image(image)`** → `clip_model.get_image_features(...)`

    * `clip_processor(images=...)`, **L2-normalize** to unit vectors.

### 3) **Vector store** (FAISS) from precomputed embeddings

```python
embeddings_array = np.array(all_embeddings)  # text + image vectors (512-d)
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,  # using our own embeddings
    metadatas=[doc.metadata for doc in all_docs]
)
```

### 4) **Retriever** (cross-modal)

```python
def retrieve_multimodal(query, k=5):
    query_embedding = embed_text(query)               # CLIP text encoder
    return vector_store.similarity_search_by_vector(  # returns text + image docs
        embedding=query_embedding, k=k
    )
```

### 5) **Multimodal prompt** for GPT-4.1

```python
def create_multimodal_message(query, retrieved_docs):
    # Text blocks → "type": "text"
    # Images     → "type": "image_url", "image_url": {"url": f"data:image/png;base64,{...}"}
    return HumanMessage(content=[ ... ])  # question + context (text + images) + instruction
```

### 6) Orchestrator

```python
def multimodal_pdf_rag_pipeline(query):
    context_docs = retrieve_multimodal(query, k=5)
    message = create_multimodal_message(query, context_docs)
    response = llm.invoke([message])  # llm = init_chat_model("openai:gpt-4.1")
    # prints which pages/images were used
    return response.content
```

### 7) Example queries (in `__main__`)

* “**What does the chart on page 1 show about revenue trends?**”
* “**Summarize the main findings from the document.**”
* “**What visual elements are present in the document?**”

---

## Setup & Run

### 1) Install

```bash
pip install pymupdf pillow torch torchvision torchaudio transformers \
            scikit-learn faiss-cpu langchain python-dotenv openai
```

### 2) Configure keys

```bash
export OPENAI_API_KEY=your_key_here
```

### 3) Prepare data

* Place a PDF (e.g., `multimodal_sample.pdf`) at the repo root or update `pdf_path` in the notebook.

### 4) Run

* Open **`multimodalopenai.ipynb`** and execute all cells. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Build-MultiModal-RAG-with-Langchain/blob/main/multimodalopenai.ipynb)
* Or adapt the final cells into a script and run:

  ```bash
  python app.py   # where app.py wraps multimodal_pdf_rag_pipeline(...)
  ```
---

## Design Choices & Trade-offs

* **One model for both modalities (CLIP)**: simple infra, true cross-modal retrieval.
* **Base64 images to the LLM**: deterministic grounding, but larger prompts than caption-only approaches.
* **Chunking**: 500/100 works for short reports; tune for longer PDFs.
* **k (top-k)**: start at 5; raise for sparse docs.

---

## Extensibility

* **Caption booster**: generate image captions and index them alongside raw image vectors (LangChain **MultiVectorRetriever**).
* **Parent context**: return entire page/section with **ParentDocumentRetriever** for more readable citations.
* **Crop-and-OCR**: for fine-grained numbers inside plots.
* **Evaluation**: integrate **Context Relevance, Groundedness, Answer Relevance**; gate low-G answers.
* **Observability**: add LangSmith/TruLens traces; log tokens & latency.

---

## Limitations

* Heavy image PDFs increase prompt size (cost).
* CLIP may miss tiny text inside figures (use crop + OCR).
* Vision LLMs can over-infer → keep **groundedness checks** and show sources.

---

## Tech Stack

* **LangChain** (chat models, `Document`, splitters)
* **PyMuPDF (fitz)** + **Pillow** (PDF + image handling)
* **Transformers** + **CLIP ViT-B/32** (unified embeddings)
* **FAISS** (vector store)
* **OpenAI GPT-4.1** (vision-capable chat model via `init_chat_model("openai:gpt-4.1")`)



