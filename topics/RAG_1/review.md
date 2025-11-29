
---

# 📘 **Foundations of Retrieval-Augmented Generation (RAG)**

---

## **1.1 Introduction to RAG**

### **What is Retrieval-Augmented Generation?**

* **RAG** is an architecture where an LLM **retrieves relevant external knowledge** from a database/vector store *before* generating an answer.
* Instead of relying solely on internal parameters, the LLM uses **augmented context** to produce accurate, grounded responses.
* It acts like: *Search Engine + LLM combined*.

### **Why RAG vs Fine-Tuning?**

| Fine-tuning                              | RAG                                  |
| ---------------------------------------- | ------------------------------------ |
| Updates model weights                    | No weight updates                    |
| Expensive (requires GPUs, data curation) | Cheap, easy to update                |
| Ingests knowledge permanently            | External knowledge updated instantly |
| Harder to maintain multiple versions     | Easy to maintain index versions      |
| Not great for large corpora              | Scales well to millions of documents |

**Key point**:
RAG is preferred when knowledge changes frequently or is too large to fit into model weights.

### **Limitations of LLMs without Retrieval**

1. **Hallucination**

   * LLMs produce confident but incorrect answers due to statistical generation without grounding.
2. **Outdated Knowledge**

   * Models trained on data months/years old → cannot answer recent or domain-specific questions.
3. **Lack of domain specificity**

   * Without retrieval, domain-heavy tasks (legal, financial, medical) become unreliable.
4. **Context-length limitations**

   * Large corpora cannot fit into prompts directly.

**RAG solves this** by giving LLMs *fresh, factual, and relevant* context.

---

## **1.2 RAG Architecture**

### ⭐ **Two-Step Pipeline: Retrieve → Generate**

#### **1. Retrieve**

* Convert query → embedding
* Search vector database for top-k relevant chunks
* Return relevant documents

#### **2. Generate**

* LLM takes *query + retrieved context*
* Produces grounded answer
* Reduces hallucinations

---

### ⭐ **The RAG Stack (Full Pipeline)**

#### **1. Indexing Pipeline (Offline Step)**

* Convert documents → cleaned text
* Chunk documents
* Generate embeddings for each chunk
* Store embeddings + metadata in vector database
* Build ANN index

#### **2. Query Embedding**

* Input query is embedded using the same embedding model as indexing
* Query vector is used for similarity search

#### **3. Retriever (Approximate Nearest Neighbor Search - ANN)**

* Finds top-k closest document vectors in the embedding space
* Methods: HNSW, IVF, PQ, Flat search

#### **4. Context Formatting**

* Selected documents are converted into a prompt
* Patterns:

  * Stuffing (simple concatenation)
  * Map-reduce (for many documents)
  * Reranking + filtering

#### **5. Generator**

* LLM takes final context and completes the generation step
* E.g., GPT-4, Claude, Llama, etc.

**The RAG pipeline works like a “semantic search layer” added in front of an LLM.**

---

## **1.3 The Retrieval Problem**

Retrieval determines **which documents** the LLM sees — the core of RAG accuracy.

### **Dense vs Sparse Retrieval**

#### **Sparse Retrieval**

* Relies on **keyword matching**, not semantics.
* Represent documents as sparse vectors based on term frequency.
* Examples:

  * **TF-IDF**
  * **BM25** (most popular)

**Pros**

* Fast
* Interpretable
* Works well when exact keywords matter

**Cons**

* Poor with semantic matches
* Misses synonyms, paraphrases

---

#### **Dense Retrieval**

* Uses embeddings for semantic meaning.
* Represent documents + queries as **dense vectors** using models like:

  * BGE
  * OpenAI embeddings
  * E5, Instructor

**Pros**

* Captures meaning, synonyms
* Excellent for long text and Q/A

**Cons**

* Needs vector DB
* More memory-heavy

---

### **Hybrid Retrieval**

* Combines dense + sparse:

  * BM25 score + embedding similarity score
* Best for real-world RAG
* Helps when:

  * Documents have rare keywords
  * Queries are partially semantic & partially lexical

---

### **Lexical Retrieval Methods**

* **TF-IDF**: counts importance of words
    * TF-IDF, or Term Frequency-Inverse Document Frequency, is a statistical method used in natural language processing to score the importance of a word in a document within a collection of documents (a corpus). It combines two metrics: Term Frequency (TF), which measures how often a word appears in a specific document, and Inverse Document Frequency (IDF), which measures how rare a word is across the entire corpus. A high TF-IDF score indicates that a word is both frequent in a particular document and uncommon across other documents, making it a good indicator of that document's unique content. 
* **BM25**:
  * BM-25, or Best Match 25, is a ranking function used by search engines to estimate how relevant a document is to a specific search query. It improves upon traditional methods like TF-IDF by accounting for factors like document length and the diminishing returns of repeated terms in a query. BM25 is widely used in modern search systems, including those based on Apache Lucene, to deliver more accurate search results
  * Most used for search engines
  * Adjusts for document length
  * Handles keyword relevance better

---

### **Embedding-Based Retrieval**

* Query and documents embedded in same vector space
* Similarity measured using cosine/dot-product
* Powers semantic search
* Used by vector DBs (Pinecone, FAISS, Weaviate, Milvus)

---

## **1.4 Vector Search Basics**

### **Embedding Spaces & Semantic Similarity**

* Embeddiungs map text into high-dimensional space (e.g., 768, 1024 dims).
* Similar meanings → vectors close together.
* Different meanings → vectors far apart.

Example:
"car" and "automobile" → close
"car" and "banana" → far

---

### **Similarity Metrics**

Used to compare vectors:

1. **Cosine Similarity (most common)**

   * Measures angle between vectors
   * Best for normalized embeddings

2. **Dot Product**

   * Faster
   * Works well with non-normalized embeddings
   * Used in many retrieval libraries

3. **Euclidean Distance**

   * Less common in RAG
   * Works for metric learning problems

---

## **ANN (Approximate Nearest Neighbor) Search Methods**

Vector databases use ANN to find nearest vectors **fast**, even in millions of items.

### **1. HNSW (Hierarchical Navigable Small World Graph)**

* Graph-based search
* Very fast (logarithmic time)
* Excellent recall
* Used by: Pinecone, Milvus, Weaviate

**Strengths**

* High accuracy
* Great for real-time search

**Weakness**

* High memory usage

---

### **2. IVF (Inverted File Index)**

* Clusters vectors into centroids
* Search only in relevant clusters
* Good for large corpora (million+ docs)

**Pros**

* Efficient for very large databases
* Good trade-off of speed/accuracy

**Cons**

* Requires training K-means
* Can fail if clustering is poor

---

### **3. Flat Search (Exact Search)**

* Compares the query vector with **every vector**
* Slow but 100% accurate

**Use Case:**

* Small datasets (~< 50k vectors)
* Highest recall required

---

### **4. Product Quantization (PQ)**

* Compresses embeddings into low-memory codes
* Enables vector search on massive scales (billions)
* Used heavily in FAISS for memory efficiency
* To perform product quantization, you first divide each high-dimensional vector into multiple sub-vectors. Next, you use a clustering algorithm like k-means to find centroids for each subspace, resulting in a set of codes for each sub-vector. Finally, you replace each sub-vector with the ID of its nearest centroid, which compresses the original vector into a short vector of IDs, saving memory

**Pros**

* Very memory efficient
* Good enough recall

**Cons**

* Lower accuracy vs HNSW
* Requires quantization training

---

---

# 📘 **2. Knowledge Sources & Chunking **

---

# **2.1 Chunking Techniques**

Chunking determines how documents are split before embedding.
Bad chunking = irrelevant retrieval = hallucinations.

---

## **1. Fixed-Size Chunking**

* The simplest technique: split every document into chunks of fixed token/character size.
* Example: **512 tokens per chunk**.

### **Pros**

* Fast and deterministic
* Works well for homogeneous text
* Easy to implement

### **Cons**

* May split sentences or concepts mid-way
* Can cause context fragmentation
* Retrieval quality degrades for unstructured text

---

## **2. Recursive Text Splitter (Structural Chunking)**

Used by LangChain, LlamaIndex.

### **How it works**

* Prefer to split along natural boundaries first:

  1. Sections
  2. Headings
  3. Paragraphs
  4. Sentences
  5. Words
* If a chunk exceeds max size → recursively split using lower-level boundaries.

### **Pros**

* Preserves natural semantic boundaries
* Better context cohesion
* Highly effective for long-form documents

### **Cons**

* Slower to process
* May create uneven chunk sizes

---

## **3. Semantic Chunking Approaches**

Chunk based on *meaning* rather than size.

### **Methods**

* Use sentence embeddings → cluster sentences
* Identify topic shifts
* Use LLM-based segmentation (prompt “split this document into coherent topics”)
* Use breakpoints based on embedding similarity thresholds

### **Pros**

* Chunks are conceptually coherent
* Best retrieval quality
* Great for technical or narrative text

### **Cons**

* Computationally expensive
* Harder to implement
* Requires fine-tuning for domain data

---

## **4. Sliding Window Approach**

* Move a window across text with fixed stride.
* Example: window size 512 tokens, stride 256 tokens.

### **Pros**

* Overcomes mid-sentence cuts
* Guarantees continuity
* Higher recall (more context variations)

### **Cons**

* Increases number of chunks (more storage cost)
* Some redundancy
* Might retrieve near-duplicate chunks

---

## **5. Overlap Strategy**

Often used with fixed-size or sliding windows.

### **Why Overlap?**

* Helps maintain context continuity
* Ensures important transitions are not lost
* Useful when chunk contains multi-paragraph logic

### **Pros**

* Minimizes semantic breaks
* Improves retrieval relevance
* Reduces hallucination from missing context

### **Cons**

* More storage → higher cost
* More duplicates retrieved
* Slightly more index compute time

---

# **2.2 Data Cleaning**

Clean data → better chunking → better retrieval.

---

## **1. Deduplication**

Eliminate repeated:

* Sentences
* Paragraphs
* Full documents
* Boilerplate blocks (e.g., website headers/footers)

### **Techniques**

* MinHash / SimHash
* Embedding similarity
* Hash-based deduplication

### **Benefits**

* Reduces index size
* Prevents repetitive or redundant retrieval
* Speeds up vector DB search

---

## **2. Boilerplate Removal**

Remove:

* Navigation menus
* Copyright notices
* Repeated footers
* HTML tags / CSS / JS
* Advertisements

### **Why?**

* These distort embeddings
* Lead to irrelevant retrieval
* Increase index cost

---

## **3. Table & Code Chunking Considerations**

### **For Tables**

* Preserve row/column relationships
* Convert into structured JSON or markdown table
* Avoid splitting mid-table
* Option: generate a “table summary” chunk

### **For Code**

* Chunk code per function/class
* Maintain syntactic boundaries
* Do not split inside loops/blocks
* Preserve indentation
* Option: store docstring + function code as single chunk

**Key rule**: *structure-first chunking for tables and code.*

---

# **2.3 Metadata Design**

Metadata is crucial for filtering, reranking, context formatting, and improving retrieval quality.

---

## **1. Basic Metadata**

For each chunk, store:

* **Title**
* **Document source URL/file**
* **Author**
* **Timestamp or publication date**
* **Section heading**
* **Tags (domain/category)**

---

## **2. Metadata Used for Filtering**

Filtering improves search precision.

### Examples:

* Only retrieve documents after **2021**
* Retrieve **policy** documents only
* Filter by **region**, **topic**, **language**

Vector DBs support metadata filtering (Pinecone, Weaviate, Milvus).

---

## **3. Metadata Used for Reranking**

Extra fields used after retrieval:

### Examples:

* **BM25 score** (for hybrid retrieval)
* **Citation count** (for research papers)
* **Document length** (penalize very small chunks)
* **Embedding similarity score**

Ranking pipeline may use:

1. BM25
2. Embedding similarity
3. Cross-encoder reranking

---

## **4. Metadata Used for Context Formatting**

* Section headers → create hierarchical prompts
* Document titles → help reduce hallucinations (“Source: ABC Manual”)
* Page numbers → helpful for citations

---

# ⭐ **Summary Cheat Sheet**

### **Chunking**

* Fixed → simple, loses semantics
* Recursive → structure-aware, best general solution
* Semantic → concept-based, expensive
* Sliding window → improves recall
* Overlap → prevents context loss, increases cost

### **Data Cleaning**

* Deduplicate aggressively
* Remove boilerplate
* Handle tables/code using structure-aware rules

### **Metadata**

* Must include title, source, timestamp
* Used for filtering, reranking, and formatting
* Better metadata → better retrieval → fewer hallucinations

---


# 📘 **3. Embeddings **

---

# **3.1 Embedding Models**

Embeddings convert text → dense numerical vectors representing semantics.
High-quality embeddings = high-quality retrieval.

---

## **① OpenAI Embeddings (text-embedding-3 small/large)**

### **text-embedding-3-large**

* High dimensional (~3k dims)
* State-of-the-art semantic performance
* Best for production-grade RAG pipelines
* Strong on multilingual + code + long-document tasks

### **text-embedding-3-small**

* Lightweight and cheaper
* Lower dimensional (~512–1024)
* Good for large-scale indexing where cost matters
* Trade-off: slightly lower accuracy

**Pros**

* Excellent quality
* Fast inference
* Great cross-domain capabilities

**Cons**

* Not open-source
* Cost at scale

---

## **② BGE (BAAI General Embedding) – Large + Variants**

Popular **open-source** family of embedding models.

Variants:

* **bge-large-en**
* **bge-base-en**
* **bge-small-en**
* **bge-m3** (multi-lingual)

### **Strengths**

* Competitive with OpenAI on many benchmarks
* Open-source & fine-tunable
* Good for industries needing on-prem deployment

### **Performance Notes**

* bge-large often outperforms smaller closed-source embeddings on niche domains
* bge-m3 is strong for multilingual retrieval

---

## **③ E5 Embeddings**

Models: `e5-large`, `e5-base`, `e5-small`

### **Key idea**

* Train models with explicit **instruction-based retrieval** format:
  *“query: ...”* vs. *“passage: ...”*

### **Pros**

* Simplifies retrieval tasks (uniform format)
* Strong performance on Q/A retrieval
* Great at multi-query/HyDE retrieval use cases

### **Cons**

* Slightly weaker on code & structured text

---

## **④ Instructor Models (Instruct-based Embeddings)**

Examples:

* `Instructor-large`
* `Instructor-base`

### **Key feature**

* Embeddings conditioned on explicit **instructions**, e.g.:
  *"Represent the scientific meaning of this sentence for retrieval"*
* More controllable embeddings

### **Pros**

* Highly robust for domain-specific tasks
* Can perform task-oriented embedding generation

### **Cons**

* Slower
* Sensitive to instruction phrasing
* Larger models → higher cost

---

# **3.2 Domain-Specific Embeddings**

General embeddings may fail for specialized vocabularies or jargon.

---

## **1. Legal Embeddings**

* Capture case law, statutes, legal terminology
* Examples:

  * Legal-BERT variants
  * Lawyer-LLaMA embeddings

### **Benefits**

* Better retrieval for legal precedence
* Understands references, citations, legal connectives

---

## **2. Finance Embeddings**

* Capture:

  * Balance sheet terminology
  * Market microstructure
  * Economic jargon
  * Company filings

### **Examples**

* FinBERT
* BloombergGPT embeddings (proprietary)

### **Benefits**

* Accurate retrieval in financial research
* Better risk assessment QA systems

---

## **3. Medical Embeddings**

* Trained on biomedical corpora:

  * PubMed
  * Clinical notes
  * Biomedical ontology

### **Examples**

* BioBERT
* PubMedBERT
* SapBERT (concept linking)

### **Benefits**

* Better retrieval for symptoms/diagnosis queries
* Better grounding with biomedical entities

---

## **When to Fine-Tune Embeddings?**

Fine-tuning is needed when:

* Domain data is very specialized
* Vocabulary not covered in general corpora
* High precision needed (e.g., legal/medical applications)
* Retrieval quality is weak using general embeddings
* Entity-level distinctions are critical
* There is a mismatch between query style and document style

### Fine-tuning helps:

* Align embeddings with domain knowledge
* Improve ranking quality
* Reduce hallucinations by better grounding

**Note:** Fine-tuning embeddings is cheaper & safer than fine-tuning LLMs.

---

# **3.3 Embedding Evaluation**

Evaluate how well embedding-based retrievers return relevant documents.

---

## **1. MTEB Benchmark Basics**

**MTEB = Massive Text Embedding Benchmark**

Covers:

* Retrieval
* Classification
* Clustering
* Reranking
* STS (Semantic Textual Similarity)
* Summarization tasks

### Metrics MTEB uses:

* Accuracy
* F1
* Spearman/Pearson correlation
* nDCG

**Purpose:** Compare embedding models across many tasks.
Open-source models like BGE/E5 often top the retrieval-heavy categories.

---

## **2. Precision@k**

Measures how many of the **top-k retrieved docs** are relevant.

### Formula:

$$
Precision@k = \frac{\text{Relevant docs in top k}}{k}
$$

### Meaning:

* High precision@k → top results are relevant
* Important for real-time user-facing search systems

---

## **3. Recall@k**

Measures how many relevant documents overall were retrieved in the top k.

### Formula:

$$
Recall@k = \frac{\text{Relevant docs retrieved}}{\text{Total relevant docs}}
$$

### Meaning:

* High recall → system finds *most of* the relevant information
* Critical for tasks where missing info is costly (legal, medical)

---

## **4. MRR (Mean Reciprocal Rank)**

Focuses on the **rank of the first correct answer**.

### Formula:

$$
MRR = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{rank_i}
$$

### Meaning:

* High MRR = relevant documents appear early in the list
* Important for chatbots & QA systems

---

# ⭐ **Summary Cheat Sheet**

### **Embedding Models**

* **OpenAI text-embedding-3:** Best performance, closed-source
* **BGE:** Best open-source
* **E5:** Query/Passage optimized
* **Instructor:** Task-conditioned embeddings

### **Domain Embeddings**

* Fine-tune when domain has special terminology
* Legal, finance, medical all benefit

### **Embedding Evaluation**

* **Precision@k** → top result quality
* **Recall@k** → coverage of relevant info
* **MRR** → how early correct result appears
* **MTEB** → global benchmarking standard

---

---

# 📘 **4. Vector Database & Retrieval Systems**

---

# **4.1 Vector DB Options**

Vector databases store embeddings and allow fast approximate nearest-neighbor (ANN) search.
Core goal → retrieve top-k semantically similar chunks with low latency.

---

## **1. Pinecone**

A fully managed, cloud-native vector database.

### **Strengths**

* High availability & low-latency global deployments
* HNSW + custom optimizations under the hood
* Easy API integration
* Fast scaling, no ops required
* Great metadata filtering support

### **Weaknesses**

* Costly at scale
* Fully proprietary (no on-prem option)

**Use-case:** Enterprise RAG, high-performance production deployments.

---

## **2. Weaviate**

Open-source + cloud-managed options.

### **Strengths**

* Built-in text vectorization modules
* Hybrid search out of the box (BM25 + vectors)
* Strong metadata filtering
* GraphQL API
* Schema-based data modeling

### **Weaknesses**

* Operational complexity for large clusters
* Indexing can be slow for massive workloads

**Use-case:** Multi-modal search, hybrid search, open-source deployments.

---

## **3. Milvus**

One of the most popular open-source vector DBs.

### **Strengths**

* Highly optimized ANN search
* Scales to billions of vectors
* Uses IVF, HNSW, PQ, GPU-accelerated indexing
* Strong open-source community (Zilliz Cloud = managed version)

### **Weaknesses**

* Complex cluster management in self-hosted mode
* Requires tuning for optimal performance

**Use-case:** Large-scale indexing (millions to billions), enterprise on-prem.

---

## **4. FAISS**

A **library**, not a database. Developed by Facebook/Meta.

### **Strengths**

* State-of-the-art ANN algorithms (Flat, IVF, PQ, HNSW)
* GPU-accelerated
* Very fast + customizable
* Perfect for building custom retrieval components

### **Weaknesses**

* No built-in persistence
* No distributed cluster support
* Requires significant engineering work

**Use-case:** Custom RAG systems, research, local prototypes.

---

## **5. Chroma**

Lightweight open-source vector store.

### **Strengths**

* Easy to use
* Perfect for personal RAG apps and quick prototypes
* Built-in persistence
* Works with Python or JS

### **Weaknesses**

* Not optimized for large-scale production
* Limited performance vs Milvus/Pinecone

**Use-case:** Prototyping, small- to medium-scale RAG apps.

---

## **6. Elasticsearch + KNN Plugin**

Adds vector search on top of Elasticsearch.

### **Strengths**

* Combines lexical BM25 + vector search
* Proven distributed search engine lineage
* Great for enterprise document search systems
* Strong metadata filtering

### **Weaknesses**

* Vector search slower vs specialized DBs
* Higher operational cost

**Use-case:**

* Hybrid search
* Enterprise search
* Systems needing both traditional search + semantic search

---

# **4.2 How Indexing Works**

Indexing = building efficient structures for fast ANN search.

---

## **1. Index Build Time**

Depends on:

* Vector DB (FAISS fastest, Pinecone managed)
* Index type (HNSW = slower to build, faster to search)
* Number of vectors
* Dimensionality (higher dims → slower)
* Hardware (CPU vs GPU)

### **Typical indexing speeds**

* FAISS IVF on GPU → millions/min
* HNSW → slower due to graph building
* PQ → requires training quantizers

---

## **2. Search Complexity**

### **Exact Search**

* Flat (L2 / cosine)
* Complexity: **O(n × d)**
* Slow at scale (>100k vectors)

### **ANN Search**

* HNSW → ~O(log n)
* IVF → ~O(√n)
* PQ → depends on quantization codebooks

**Trade-off:**
Speed ↑ → Recall ↓

---

## **3. Memory Considerations**

### Factors affecting memory:

* Vector size (dimensionality)
* Data type (float32 vs float16)
* Index type
* Metadata stored

### Notes:

* HNSW uses ~2–5× vector size due to graph edges
* PQ compresses vectors up to 4–10×
* FAISS Flat index stores raw embeddings → high memory consumption

---

# **4.3 Filtering & Metadata Search**

Metadata enables restricting search to relevant subsets of vectors.

---

## **1. Hybrid Retrieval**

Combines:

* **Sparse retrieval** (BM25)
* **Dense retrieval** (embeddings)

Framework:

1. Compute BM25 score
2. Compute vector similarity
3. Combine scores (weighted sum or reranking)

### Benefits:

* Handles keyword-heavy queries
* Recovers cases where dense retrieval fails
* Best overall accuracy for enterprise RAG

---

## **2. Metadata Filtering Strategies**

### **Date Filtering**

* Useful for time-sensitive domains (news, regulations, product updates)
  Example:
  Retrieve only documents after *2022-01-01*.

### **Topic Filtering**

* Pre-assign document tags/labels
* Filter by: product, domain, category, legal section, department, etc.

### **Other Filters**

* Author
* File type
* Language
* Version
* Access permissions (multi-tenant RAG)

**Note:** Good metadata dramatically improves RAG groundedness.

---

# **4.4 Rerankers**

Rerankers re-order retrieved chunks using more accurate (but slower) models.

---

## **1. Cross-Encoder Reranking**

Use a transformer that takes both **query + document** together and predicts relevance.

### **Models**

* **BGE-reranker (best open-source)**
* **Cohere Rerank (state-of-the-art, proprietary)**

### Strengths

* Significantly improves retrieval quality
* Captures fine-grained semantic alignment
* Reduces hallucinations by prioritizing best chunks

### Weaknesses

* Slower (must run full transformer per candidate)
* Needs batching for performance

---

## **2. Multi-Stage Retrieval Pipeline**

A common strategy in production:

### **Stage 1 — Fast Retrieval**

* ANN vector search (top-50 / top-100)
* BM25 + vector hybrid retrieval

### **Stage 2 — Filter**

* Metadata filters
* Deduplicate nearly identical chunks

### **Stage 3 — Rerank (Cross-Encoder)**

* Use BGE-reranker or Cohere Rerank
* Pick top-3 to top-10 most relevant

### **Stage 4 — LLM Input**

* Final context chunks passed to LLM

**Effect:**

* High precision
* Fewer irrelevant contexts
* Lower LLM cost
* More grounded responses

---

# ⭐ **Summary Cheat Sheet**

### **Vector DB Options**

* **Pinecone:** Best managed solution
* **Weaviate:** Open-source, hybrid search
* **Milvus:** Massive scale, GPU support
* **FAISS:** Library for custom pipelines
* **Chroma:** Best for small apps
* **Elasticsearch:** Hybrid enterprise search

### **Indexing**

* HNSW fastest search, slow build
* IVF for massive datasets
* PQ for compression
* Flat for exact search (slow)

### **Filtering**

* Use metadata: date, topic, author
* Hybrid (BM25 + embeddings) improves accuracy

### **Reranking**

* Cross-encoders dramatically improve top-k quality
* Multi-stage retrieval is the production standard

---
