
---

# 📘 **5. RAG Query Pipeline**

---

# **5.1 Query Reformulation**

Query reformulation improves the **quality of retrieved documents**, which directly impacts the quality of RAG answers.

---

## **1. Query Rewriting → Better Retrieval**

Rewriting the user's query into a **more retrieval-friendly format**.

### **Why?**

* User queries are often ambiguous
* Embeddings may not capture intent clearly
* Reformulated queries increase recall and relevance

### **Techniques**

* Use an LLM to rewrite:
  *“Rewrite this query to optimize for semantic retrieval.”*
* Expand keywords
* Clarify context
* Use synonyms or related concepts
* Rephrase into a complete sentence

### Example

Input: “Tesla revenue last year?”
Rewritten: “What was Tesla’s total annual revenue in the previous fiscal year?”

---

## **2. Multi-Hop Question Decomposition**

Break down complex, multi-part questions into simpler sub-queries.

### Why?

* RAG struggles with long, multi-step logic
* Retrieval often fails for multi-hop reasoning

### Approach

* Use LLM to decompose:
  *“Decompose this question into atomic queries required for retrieval.”*
* Retrieve for each sub-question
* Combine results

### Example

Query: “What company owns Instagram and where is its HQ?”
Decomposed:

1. “Who owns Instagram?”
2. “Where is Meta's headquarters?”

---

## **3. Generated Queries (HyDE)**

**HyDE = Hypothetical Document Embeddings**

Instead of embedding the raw query:

1. Generate a **hypothetical answer passage**
2. Embed that passage
3. Use it for retrieval

### Why it works

* Complex queries produce noisy embeddings
* Hypothetical answer → richer context → better retrieval
* Great for Q/A tasks and difficult reasoning queries

### Example

Query: “Why did the dinosaurs go extinct?”
HyDE generates a hypothetical paragraph about asteroid impact.
Embedding this paragraph yields better retrieval.

---

# **5.2 Retrieval Strategies**

Effective retrieval determines the usefulness of the final generated answer.

---

## **1. Top-k Selection**

Retrieve the **k most similar** documents.

### Pros

* Simple
* Fast

### Cons

* Redundancy: many chunks may be similar
* Missing unique or diverse contexts
* Can lead to hallucinations if k is too small

**Typical values:** k = 3, 5, 10, or 20 (before reranking)

---

## **2. Max Marginal Relevance (MMR)**

Balances **relevance** and **diversity** of retrieved chunks.

### Formula

$$
MMR = \lambda \cdot relevance - (1 - \lambda) \cdot redundancy
$$

### Why it helps

* Prevents selecting near-duplicate chunks
* Improves coverage of different subtopics

### Common λ values

* λ ≈ 0.5 (balanced)

---

## **3. Adaptive Retrieval**

Retrieve variable number of chunks based on:

* Query complexity
* Document density
* Confidence score
* Reranker scores

### Examples

* Simple fact queries → 1–2 chunks
* Multi-hop or summarization queries → 5–10 chunks
* Low similarity scores → fallback to hybrid retrieval

### Benefits

* More efficient
* More accurate
* Reduces irrelevant context

---

## **4. Self-Query Retrieval**

The LLM creates the **metadata filters + query vector** itself.

### Example

Input: “Find regulations about data privacy after 2020.”

LLM generates:

* Query embedding: “data privacy regulations”
* Metadata filter: `date >= 2020`
* Topic filter: “legal”

### Benefits

* Automates complex filtering
* Great for structured document corpora

Framework support:

* LangChain
* LlamaIndex

---

# **5.3 Prompt Engineering for RAG**

Prompting determines **how the LLM uses retrieved context**.

---

## **1. Stuffing**

The simplest method:
**Concatenate all retrieved documents into the prompt.**

### Pros

* Easy
* Works for small contexts

### Cons

* Doesn’t scale with long documents
* Can confuse the model if context is unstructured
* Prone to prompt overflow

---

## **2. Map-Reduce Prompting**

A technique for large document sets.

### **Map Stage**

LLM processes each chunk independently:

* Summarize
* Extract facts
* Answer sub-question

### **Reduce Stage**

LLM combines all map-stage outputs.

### Benefits

* Scales to 100+ chunks
* Great for summarization and research tasks

### Use cases

* Report generation
* Analysis across multiple documents

---

## **3. Refinement Chain**

Iteratively refine answers with retrieved context.

### Procedure

1. Take initial answer
2. Add next chunk
3. Ask LLM to refine
4. Repeat

### Benefits

* Handles long, multi-hop reasoning
* Produces higher-quality answers
* Less prone to hallucinations

LlamaIndex implements this natively.

---

## **4. Context-Window Optimization**

Techniques to maximize useful content within LLM’s token window.

### Methods

* Remove redundant or similar chunks
* Use aggressive chunk deduplication
* Shorten documents via compression or summarization
* Use rerankers to choose best top-3 or top-5 chunks
* Use long-context LLMs for high recall tasks

### Best practices

* Use cross-encoder reranker before stuffing
* Keep final context < 50% of available window
* Prefer structured formatting (bullet points, tables)

---

# ⭐ **Summary Cheat Sheet**

### **Query Reformulation**

* Query rewriting → clearer semantics
* Multi-hop decomposition → better coverage
* HyDE → generates hypothetical answers → better embeddings

### **Retrieval Strategies**

* Top-k: simple, baseline
* MMR: increases diversity
* Adaptive retrieval: dynamic k based on complexity
* Self-query: LLM generates metadata filters

### **Prompting Patterns**

* Stuffing → simple, small contexts
* Map-reduce → large-scale document tasks
* Refinement → iterative improvement
* Context-window optimization → reduce overflow + redundancy

---

---

# 📘 **6. Advanced RAG **

---

# **6.1 RAG Techniques**

These are modern RAG variations that improve retrieval quality, reasoning, and scalability.

---

## **1. Self-RAG**

A self-reflective RAG system where the LLM **evaluates and critiques** retrieved documents before using them.

### **How it works**

1. Retrieve documents
2. LLM scores them for:

   * Relevance
   * Helpfulness
   * Conflicts
3. LLM discards or re-ranks documents
4. LLM answers the question using curated context
5. LLM evaluates its own answer for groundedness

### **Benefits**

* Reduces hallucinations
* Improves precision by removing irrelevant chunks
* Self-correction behavior without expensive rerankers

### **Use-case**

Q/A tasks where correctness is critical (legal, finance, compliance).

---

## **2. GraphRAG (Retrieval over Knowledge Graphs)**

GraphRAG converts raw documents into a **knowledge graph** (nodes = concepts, edges = relationships).

### **How retrieval works**

1. Extract entities & relations using LLMs
2. Build a graph structure
3. Query is mapped to graph nodes
4. Retrieve subgraphs or neighborhoods
5. Convert graph results into prompt context

### **Benefits**

* Handles multi-hop reasoning
* Connects facts hidden across multiple documents
* Avoids repetition and redundancy in chunk-level retrieval

### **Use-case**

* Research intelligence
* Enterprise knowledge management
* Docs with high inter-connectedness (legal, medical)

---

## **3. Agentic RAG**

LLM acts as an **agent** that:

* Plans actions
* Retrieves iteratively
* Refines queries
* Combines multiple retrieval strategies
* Performs tool calls (search DB, filter, compute)

### **Workflow**

1. Analyze user query
2. Generate plan
3. Run retrieval steps iteratively
4. Invoke external tools (SQL, vector search, calculators)
5. Produce grounded final answer

### **Benefits**

* Handles complex tasks
* Multi-step reasoning
* Dynamic retrieval

### **Frameworks**

* LangGraph
* LlamaIndex Agents
* OpenAI Assistants API (multi-tool)

---

## **4. Multi-Document Summarization RAG**

Instead of retrieving top-k chunks directly → retrieve **many** chunks and perform hierarchical summarization.

### **Patterns**

* Map-Reduce summarization
* Tree-based summarization
* LLM hierarchical compression

### **Benefits**

* Handles 100+ documents
* Lower hallucination rate for multi-source tasks
* Produces consistent, global summaries

**Use-case:** internal research reports, literature reviews.

---

## **5. RAG-Fusion (Query Aggregation)**

Inspired by “Fusion-in-Decoder” from search literature.

### **How it works**

1. Generate multiple query variants
2. Retrieve documents for each variant
3. Fuse all retrieval results (e.g., reciprocal rank fusion)
4. Rerank and select top documents

### **Benefits**

* Dramatically increases recall
* Makes retrieval robust to phrasing changes
* Useful when user queries are ambiguous

### **Example**

User query:
"energy consumption reduction methods"

Generated queries:

* “how to reduce energy use”
* “energy efficiency techniques”
* “reduce electricity consumption strategies”

Aggregate retrieval → richer context.

---

## **6. ColBERT-Style Late Interaction Retrieval**

A **late interaction** model where:

* Documents are broken into token-level embeddings
* Query tokens interact with document tokens
* Scoring uses partial similarity rather than full vector comparison

### **Benefits**

* High precision for long docs
* Great for fine-grained semantic matching
* Outperforms vanilla dense retrieval on many tasks

### **Cons**

* Heavier compute
* Requires specialized infrastructure

### **Use-case**

Large-scale retrieval where accuracy is critical (research, legal search).

---

# **6.2 RAG Optimization**

Techniques to improve retrieval efficiency and answer quality.

---

## **1. Ranking Losses**

Used when fine-tuning embedding models.

### **Common losses**

1. **Contrastive loss**
2. **Triplet loss**
3. **In-batch negatives**
4. **Margin ranking loss**

### **Goal**

Train embeddings so that:

* Relevant docs are closer
* Irrelevant docs are farther

**Effect:** Improves retrieval precision and recall.

---

## **2. Context Deduplication**

Multiple retrieved chunks may contain overlapping or identical sentences.

### **Why it matters**

* Reduces wasted context window
* Reduces LLM confusion
* Improves groundedness

### **Techniques**

* Sentence-level deduplication
* Embedding similarity pruning
* Hash-based deduplication

---

## **3. Tokenizer-Aware Chunking**

Chunk boundaries aligned with the tokenizer reduce:

* Mid-sentence breaks
* Context loss
* Embedding discontinuity

### **Methods**

* Ensure chunks end at sentence boundaries
* Ensure each chunk fits model’s token window
* Overlap chunks to preserve semantic flow
* Avoid splitting code blocks, tables

### **Outcome**

Better semantic coherence → better retrieval → fewer hallucinations.

---

# **6.3 Evaluation**

Evaluation in RAG is **multi-dimensional**: correctness, grounding, relevance, factuality, coherence.

---

## **1. Hallucination Tests**

Measure how often the model:

* Produces unsupported statements
* Makes up facts
* Misinterprets retrieved context

### **Approaches**

* Compare output vs gold reference
* Ask a second LLM:
  *“Is this statement grounded in the provided context?”*
* Detect contradictions

---

## **2. Groundedness Scoring**

Measures how much of the answer is based on given context.

### **Common approach**

* LLM-as-judge checks each sentence:
  *Supported / Unsupported / Unrelated*
* Count supported vs total sentences

### **Metric**

$$
\text{Groundedness Score} = \frac{\text{Supported Sentences}}{\text{Total Sentences}}
$$

---

## **3. ROUGE, BLEU, Factuality Checks**

These metrics are not perfect for RAG but still useful:

### **ROUGE**

* Measures overlap between generated text and ground truth
* Good for summarization tasks

### **BLEU**

* Measures n-gram precision
* Better for structured answers

### **Factuality checks**

* Entity-level correctness
* Relation correctness
* Numerical correctness

Tools like **FactScore**, **TruthfulQA**, or LLM-based grading used.

---

## **4. LLM-as-a-Judge Evaluation Pipeline**

Modern evaluation technique.

### **Pipeline**

1. Provide **context**, **query**, and **answer** to a strong LLM (GPT-4/5, Claude).
2. Ask it to rate:

   * Relevance
   * Groundedness
   * Completeness
   * Hallucination likelihood
   * Citation correctness
3. Aggregate scores.
4. Optionally run multiple judges and average scores.

### **Benefits**

* Fast, automated, scalable
* More accurate than traditional metrics alone
* Works well for open-ended Q/A

---

# ⭐ **Summary Cheat Sheet**

### **Advanced RAG Techniques**

* **Self-RAG**: Reflective retrieval + self-critique
* **GraphRAG**: Knowledge graph-based retrieval
* **Agentic RAG**: Multi-step tool use + planning
* **Multi-doc RAG**: Summarization across many docs
* **RAG-Fusion**: Multi-query aggregation
* **ColBERT**: Late interaction token-level retrieval

### **Optimization**

* Ranking losses improve embeddings
* Deduplication reduces redundancy
* Tokenizer-aware chunking → better semantic coherence

### **Evaluation**

* Hallucination tests
* Groundedness scoring
* ROUGE/BLEU/factuality
* LLM-as-a-judge pipelines (industry standard)

---

---

# 📘 **7. System Design for RAG**

---

# **7.1 Components in Production RAG**

Modern RAG systems are multi-service architectures. Below are the core components.

---

## **1. Indexing Service**

Responsible for building and maintaining vector indexes.

### Responsibilities:

* Read raw documents
* Clean, chunk, and embed text
* Deduplicate content
* Store vectors + metadata in vector DB
* Handle re-indexing and updates

### Key properties:

* Scalable
* Asynchronous
* Supports incremental indexing

---

## **2. Document Pipeline**

Transforms raw source data into RAG-ready chunks.

### Steps:

1. **Ingestion**

   * PDFs, webpages, internal knowledge bases, SQL tables
2. **Parsing**

   * OCR, HTML extraction, text cleaning
3. **Chunking**

   * Recursive or semantic chunking
4. **Metadata extraction**

   * Title, author, date, section
5. **Embedding**

   * Use embedding model
6. **Vector DB write**

   * Store embeddings + metadata

### Properties:

* Highly parallel
* Fault tolerant
* Supports batch updates or streaming

---

## **3. LLM Inference Gateway**

The service that manages all interactions with the LLM.

### Responsibilities:

* Receives user query
* Performs query rewriting (HyDE, multi-query)
* Calls retriever
* Formats context into prompt
* Calls LLM for generation
* Applies output post-processing (citation extraction, validation)

### Good gateway characteristics:

* Rate limiting
* Batching for cost savings
* Fast fallback models based on query complexity
* Monitoring for hallucinations and groundedness

---

## **4. Rerankers**

Cross-encoder or task-specific rerankers sit between retrieval and generation.

### Purpose:

* Improve relevance of top-k retrieved chunks
* Filter out noise and duplicates

Common rerankers:

* BGE-Reranker
* Cohere Rerank
* ColBERT-lite reranking

---

## **5. Cache Layer**

Caches reduce redundant work and improve latency.

### Types:

* **Query-result cache** (semantic + lexical)
* **Embedding cache**
* **Reranker cache**
* **LLM response cache**

### Benefits:

* Lower cost
* Lower latency
* Handles repetitive enterprise queries

---

## **6. Monitoring (Latency, Recall, Etc.)**

Production RAG must be monitored like any distributed system.

### Metrics:

**Retrieval metrics:**

* Recall@k
* Precision@k
* Avg retrieved chunk duplication rate

**Latency metrics:**

* Embedding model latency
* Vector DB latency
* Reranker latency
* LLM latency

**Quality metrics:**

* Groundedness score
* Hallucination probability
* Conversation success rate

**System health:**

* Index freshness
* Memory usage
* Disk usage
* QPS (Queries Per Second)

---

# **7.2 Performance Considerations**

Scaling RAG in production requires deep optimization.

---

## **1. Latency Budgets**

The total time breakdown typically looks like:

| Component       | Typical Latency |
| --------------- | --------------- |
| Query Embedding | 5–20 ms         |
| Vector Search   | 10–40 ms        |
| Reranker        | 30–150 ms       |
| LLM Generation  | 150–900 ms      |
| End-to-end RAG  | 300–1500 ms     |

Latency optimization must:

* Reduce k
* Use faster embeddings
* Use distilled LLMs for simple queries
* Cache aggressively

---

## **2. GPU vs CPU for Retrieval**

### CPU retrieval:

* Best for HNSW
* Cheaper
* Good for up to millions of vectors
* Great for Pinecone/Weaviate

### GPU retrieval:

* Best for FAISS IVF-PQ, flat search
* Needed for billion-scale search
* Good for high QPS + low latency workloads

### Rule of thumb:

* **CPU** for semantic QA
* **GPU** for massive-scale search or PQ-indexed data

---

## **3. Sharding & Replication of Vector DB**

### Sharding:

* Splits index into multiple nodes
* Required when vectors exceed node memory
* Query must be evaluated on all shards → aggregator returns top-k

### Replication:

* Ensures high availability
* Parallelism for high read throughput

### Problems:

* Cross-shard consistency
* Higher cost
* Shard imbalance (skewed corpus)

---

## **4. Batch vs Streaming Pipelines**

### Batch indexing:

* Daily/weekly ingestion
* Good for static or slow-changing data
* Cheaper but less fresh

### Streaming indexing:

* Ingest documents in near real-time
* Event-driven (Kafka → chunker → indexer)
* Necessary for:

  * News articles
  * Financial reports
  * Support ticket systems
  * Real-time analytics

Trade-offs:

* More complex
* More expensive
* Better user experience

---

# **7.3 Failure Cases**

Understanding failure modes is crucial for interviews and production.

---

## **1. Retrieval Collapse**

System returns irrelevant documents.

### Causes:

* Poor chunking
* Bad embeddings
* Encoding mismatch (different embedding models)
* Vector DB overload
* Query too short or ambiguous

### Fixes:

* Query rewriting
* Rerankers
* Hybrid search
* Better chunking strategy

---

## **2. Embedding Drift**

When embeddings become inconsistent across versions.

### Causes:

* Upgrading embedding models
* Changing chunking strategy
* Retraining without re-indexing

### Effects:

* Decline in recall
* Vector DB mismatch
* Inconsistent retrieval

### Fix:

* Re-index entire corpus after embedding model change.

---

## **3. Vector DB Corruption**

### Causes:

* Incomplete writes
* Crash during indexing
* Incorrect metadata
* Shard failures

### Fix:

* Maintain backups
* Use replicated clusters
* Monitor index integrity

---

## **4. Context Overflow**

Too many tokens push past the LLM's context window.

### Symptoms:

* Important chunks are truncated
* Model hallucinates missing details
* Prompt becomes unstructured

### Fix:

* Aggressive reranking
* Context compression
* Map-reduce summarization
* Multi-query fusion
* Token-aware chunk selection

---

# **7.4 Interview-Ready RAG Design Problems**

Prepare structured answers for these common interview prompts.

---

## **1. “Design a RAG-based QA system”**

### Key points:

* Document ingestion pipeline
* Chunking + embeddings
* Vector search + metadata filtering
* Reranking (BGE/ColBERT)
* LLM inference gateway
* Caching layer
* Monitoring & evaluation (recall, groundedness)
* Protection against hallucinations
* Privacy + access control

---

## **2. “Design enterprise search for millions of documents”**

Discuss:

* Sharded vector DB
* Hybrid BM25 + dense retrieval
* Caching for popular queries
* Reranker pipeline
* Multi-tenant access control
* Horizontal scaling
* Index freshness strategy (streaming ingestion)

---

## **3. “Design RAG for multilingual content”**

Mention:

* Multilingual embedding models (bge-m3, LaBSE)
* Language detection + routing
* Locale-based metadata filtering
* Cross-language reranking
* Per-language LLM models if needed

---

## **4. “Design a RAG for legal document analysis”**

Include:

* Legal-specific embeddings (Legal-BERT)
* Long-document chunking (semantic)
* Citation extraction
* Fact consistency checks
* Strict hallucination guardrails
* GraphRAG for linking cases & laws

---

## **5. “How do you prevent hallucinations in RAG?”**

Answer with:

1. Better retrieval (rerankers, hybrid search, fusion)
2. Context deduplication
3. Self-RAG reflection
4. Groundedness scoring
5. Retrieval confidence thresholds
6. LLM guardrails in prompt
7. Output validation layer (“cite your sources”)

---

# ⭐ **Summary Cheat Sheet**

### **Components**

* Indexing service
* Document pipeline
* LLM gateway
* Rerankers
* Cache
* Monitoring

### **Performance**

* Latency budget breakdown
* CPU vs GPU retrieval trade-offs
* Sharding + replication
* Batch vs streaming ingestion

### **Failure Cases**

* Retrieval collapse
* Embedding drift
* DB corruption
* Context overflow

### **Interview Design Problems**

* RAG QA
* Enterprise search
* Multilingual RAG
* Legal RAG
* Hallucination prevention

---
