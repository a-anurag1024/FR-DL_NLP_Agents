**🧠 Foundations of Sequence Modeling**
---

### 🌟 Concept Overview

**Goal:** Model sequential dependencies in text/speech (order matters!)
**Shift:** From **count-based (n-gram)** to **learned neural sequence models (RNNs, LSTMs, etc.)**

---

### 🔹 1. Traditional Models — n-gram Approach

* **Idea:** Predict next word using previous (n–1) words
  → P(wₜ | wₜ₋₁, wₜ₋₂, …, wₜ₋ₙ₊₁)
* **Pros:** Simple, interpretable
* **Cons:**

  * Fixed-length context (no long-term dependencies)
  * Data sparsity → unseen n-grams have zero probability
  * Need smoothing (Laplace, Kneser-Ney)
* **Transition Motivation:** Fails to scale for long sequences → Neural sequence models introduced

---

### 🔹 2. Neural Sequence Models — RNN Family

**Key Concept:** Hidden state captures “memory” of previous inputs

#### 🧩 Vanilla RNN

* **Equation:** hₜ = f(Wₕₓxₜ + Wₕₕhₜ₋₁ + bₕ)
* **Challenge:** Vanishing/exploding gradients

  * Long sequences → earlier timesteps have negligible influence

#### ⚙️ Deep & Bi-Directional RNNs

* **Deep RNN:** Multiple stacked RNN layers → hierarchical features
* **Bi-Directional RNN:**

  * Forward + backward context
  * Useful for tasks needing both past and future info (e.g. NER)

#### 🔄 GRU (Gated Recurrent Unit)

* **Simplified LSTM:** Combines forget & input gates into **update gate**
* **Fewer parameters**, faster training

#### 🧠 LSTM (Long Short-Term Memory)

* **Solves vanishing gradient problem** via **cell state (Cₜ)**
* Gates:

  * **Forget gate (fₜ)** → What to remove
  * **Input gate (iₜ)** → What to add
  * **Output gate (oₜ)** → What to expose
* Enables **long-term dependency capture**

---

### 🔹 3. RNNs and the Vanishing Gradient Problem

* **Reason:** Repeated multiplication of small weights < 1
* **Effect:** Early timesteps lose gradient contribution → memory fades
* **Fixes:**

  * Use gated units (LSTM/GRU)
  * Gradient clipping
  * Layer normalization
  * Residual connections in deeper RNNs

---

### 🔹 4. Sequence Model vs Attention Model

| Aspect                  | Sequence (RNN-based)              | Attention-based           |
| :---------------------- | :-------------------------------- | :------------------------ |
| **Dependency modeling** | Sequential (one step at a time)   | Parallel (global context) |
| **Memory capacity**     | Limited (hidden state bottleneck) | Full access to all tokens |
| **Speed**               | Slow (non-parallel)               | Fast (parallelizable)     |
| **Examples**            | LSTM, GRU                         | Transformer, BERT, GPT    |

**Key Insight:**
RNNs encode context into a single hidden vector → information bottleneck.
Attention lets the model **directly “attend”** to all past tokens simultaneously.

---

### 🔹 5. Application Highlight — Named Entity Recognition (NER)

* **Goal:** Identify entities like *person*, *location*, *organization*
* **Model Setup:**

  * Input: sequence of words
  * Output: label per word (BIO tagging: B = Begin, I = Inside, O = Outside)
* **Typical Model:**

  * Bi-LSTM (captures both directions)
  * * CRF layer (Conditional Random Field) for sequence-level labeling consistency
* **Evaluation Metrics:** Precision, Recall, F1-score

---

### 🔹 6. Key Takeaways

✅ Neural sequence models overcome fixed-context limits of n-grams
✅ RNNs introduced recurrence → memory of past inputs
✅ LSTMs/GRUs solve vanishing gradients via gating mechanisms
✅ Bi-directional models help in context-rich tasks (e.g., NER)
✅ Attention models supersede RNNs in efficiency and global context modeling

---

**🧭 Quick Mnemonics:**

* **N-gram → RNN → LSTM → Attention → Transformer**
* **Gates (LSTM):** Forget what’s unnecessary, Input new info, Output meaningful context.
* **NER:** Bi-LSTM + CRF = context-aware sequence labeling

---
**⚡ Attention Mechanisms & Transition to Transformers**
*(The revolution that replaced recurrence with attention and gave rise to Transformers)*

---

### 🌍 Concept Overview

**Goal:** Enable models to capture **long-range dependencies** and **contextual relationships** *without recurrence*.
**Breakthrough:** Attention allows **direct access** to all input positions simultaneously → faster, more global understanding.

---

### 🔹 1. The Attention Concept

**Core Idea:** Let the model **“focus”** on relevant parts of the input sequence when producing an output.

* Each output token is computed as a **weighted sum** of all input tokens.
* Weights (attention scores) indicate **importance**.

#### 🧮 Scaled Dot-Product Attention

Formula:
**Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V**

* **Q (Query):** What we’re looking for (current token)
* **K (Key):** What each input token represents
* **V (Value):** The information each token holds
* **Scaling by √dₖ:** Prevents large dot products from saturating softmax
* **Intuition:** Similarity(Q, K) → attention strength

---

### 🔹 2. Types of Attention

Different variants for different purposes in model architectures:

#### 🔸 Self-Attention

* **Query, Key, Value come from the same sequence.**
* Each token attends to **all tokens (including itself)**.
* Captures context relationships within a sentence.
* Example: *“The animal didn’t cross because it was too tired.”* → “it” attends to “animal.”

#### 🔸 Cross-Attention

* Used in **Encoder–Decoder** models (e.g., Translation).
* **Query** = decoder’s current hidden state
* **Key/Value** = encoder outputs
* Lets the decoder “look back” at the source sequence.

#### 🔸 Masked Self-Attention

* Used in **autoregressive models** (e.g., GPT).
* Prevents looking ahead → ensures causal (left-to-right) prediction.

#### 🔸 Multi-Head Attention

* Multiple attention heads = multiple representation subspaces.
* Each head learns different relationships (syntax, semantics, etc.).
* Final outputs are concatenated and projected → richer context representation.

**Analogy:** Like multiple readers focusing on different parts of the same paragraph.

---

### 🔹 3. Position Encoding

**Problem:** Attention has no notion of word order (no recurrence).
**Solution:** Add **positional information** to token embeddings.

* **Sinusoidal Encoding:**

  * Fixed patterns (sin & cos) of different frequencies
  * Allows model to infer relative positions via continuous signals
* **Learned Positional Embeddings:**

  * Parameters learned during training (e.g., BERT)

**Intuition:**
Encodes *“where”* each token is in sequence → gives attention models a sense of order.

---

### 🔹 4. RNNs vs Transformers

| Aspect                | RNNs                             | Transformers                       |
| :-------------------- | :------------------------------- | :--------------------------------- |
| **Processing**        | Sequential (one token at a time) | Parallel (entire sequence at once) |
| **Memory**            | Hidden state bottleneck          | Global context via attention       |
| **Long Dependencies** | Hard to capture                  | Easily modeled                     |
| **Training Time**     | Slow                             | Fast (GPU-friendly)                |
| **Architecture**      | Recurrent layers                 | Stacked attention blocks           |

**Core Transition:**
Transformers removed recurrence → enabling parallelism, scalability, and massive pretraining.

---

### 🔹 5. Transformer Architecture Overview

**Encoder–Decoder Design** (e.g., in original *Vaswani et al., 2017*):

* **Encoder:** Stack of self-attention + feed-forward layers
* **Decoder:** Self-attention + cross-attention + feed-forward layers
* **Residual connections + LayerNorm** → stabilize training

---

### 🔹 6. Landmark Transformer Models

**🚀 GPT (Generative Pre-trained Transformer)**

* **Type:** Decoder-only
* **Objective:** Next-token prediction (causal LM)
* **Attention:** Masked self-attention
* **Use:** Text generation, chatbots, code models (GPT-4, GPT-5)

**🧩 BERT (Bidirectional Encoder Representations from Transformers)**

* **Type:** Encoder-only
* **Objective:** Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
* **Attention:** Bidirectional self-attention
* **Use:** Text understanding, classification, QA

**🔄 T5 (Text-to-Text Transfer Transformer)**

* **Type:** Full Encoder–Decoder
* **Objective:** Unified text-to-text task framing (“translate everything to text”)
* **Use:** Summarization, translation, question answering

---

### 🔹 7. Key Insights & Review Notes

✅ Attention replaces recurrence with *direct relevance weighting*
✅ Multi-head attention learns diverse context relationships
✅ Positional encoding gives order sense
✅ Transformers scale efficiently with parallelization
✅ GPT = Generation | BERT = Understanding | T5 = Unified Text2Text
✅ RNNs = sequential, Transformers = fully parallel

---

### 🧭 Quick Mnemonics

* **“QKV Rule” → Query, Key, Value → Attention core.**
* **“Self” = same sequence | “Cross” = between encoder & decoder.**
* **“Mask” = future-blocking for prediction.**
* **“Multi-head” = multiple perspectives.**
* **“Transformers” = attention + position + feed-forward.**

---
**🎯 Advanced Training & Decoding Techniques**
*(Training tricks, decoding strategies, and evaluation metrics that shape sequence-to-sequence NLP systems)*

---

### 🌍 Concept Overview

**Goal:** Improve **training stability**, **generation quality**, and **evaluation reliability** in sequence models (especially Neural Machine Translation & Text Generation).
**Scope:** Covers how models learn (training dynamics), how they generate text (decoding), and how we measure their quality (evaluation).

---

### 🔹 1. Neural Machine Translation (NMT) with Attention

**Core Idea:** Translate a source sequence into a target sequence using an **encoder–decoder** setup guided by **attention**.

* **Encoder:** Encodes source tokens → hidden representations
* **Attention Mechanism:** Learns alignment weights between source & target tokens
* **Decoder:** Generates target tokens while “attending” to relevant source words

**Advantages over vanilla seq2seq:**
✅ Handles long sequences better
✅ Dynamic focus on relevant words per step
✅ Enables interpretable alignment visualization

---

### 🔹 2. Pre-Attention vs Post-Attention Decoding

**🧩 Pre-Attention Decoding**

* Decoder uses attention **before** combining with previous hidden state.
* Focuses early on context → helpful for early alignment decisions.

**🔄 Post-Attention Decoding**

* Decoder first computes hidden state, then applies attention.
* Context integrated later → allows more refined, top-down context control.

**💡 Insight:**
Both are architectural design choices in encoder-decoder models; they influence **information flow** and **training stability**.

---

### 🔹 3. Teacher Forcing

**Definition:** During training, feed the **ground-truth token** as the next input instead of the model’s prediction.

**Pros:**
✅ Speeds up convergence
✅ Stabilizes early training

**Cons:**
❌ Exposure bias — model never learns to recover from its own mistakes at inference time.

**Fixes / Alternatives:**

* **Scheduled Sampling:** Gradually replace true tokens with model predictions
* **Professor Forcing:** Regularizes hidden dynamics between training & inference

---

### 🔹 4. Decoding Strategies — How Models Generate Text

Different methods to **sample** or **select** the next token during inference.

#### 🔸 Random Sampling

* Picks tokens based on probability distribution (purely random).
* High diversity, low coherence.

#### 🔸 Greedy Decoding

* Always pick the token with the highest probability (argmax).
* Simple but often **repetitive** and lacks diversity.

#### 🔸 Temperature Sampling

* Adjusts “creativity” by scaling logits before softmax:

  * **Low T (<1):** More deterministic, sharper distribution
  * **High T (>1):** More diverse, flatter distribution

#### 🔸 Beam Search

* Keeps **k** best partial sequences (beams) at each step.
* Expands them until completion, picks best-scoring final sequence.
* Balances between greedy (k=1) and exhaustive search.

#### 🔸 Top-k Sampling

* Restricts choices to the **top-k** most probable tokens, renormalizes distribution.
* Prevents extremely low-probability tokens.

#### 🔸 Top-p (Nucleus) Sampling

* Chooses smallest token set whose cumulative probability ≥ **p** (e.g., 0.9).
* Adapts cutoff dynamically based on distribution shape.

**Hierarchy of Control:**
Greedy < Beam < Top-k < Top-p < Random (increasing diversity, decreasing determinism)

---

### 🔹 5. Problems with Beam Search

⚠️ Common failure modes in text generation:

* **Lack of diversity:** All beams converge to similar outputs.
* **Length bias:** Prefers shorter sequences (due to cumulative probability drop).
* **Overconfidence:** Amplifies small early mistakes.
* **Solution Approaches:**

  * Length normalization
  * Diverse beam search
  * Penalizing repeated n-grams

---

### 🔹 6. Minimum Bayes Risk (MBR) Decoding

**Goal:** Select output minimizing **expected loss** rather than maximizing probability.

**Formula:**
ŷ = argmin_y′ E_y[L(y, y′)]
→ choose the output most similar to *many good hypotheses* instead of the single highest-probability one.

**Benefits:**
✅ Improves faithfulness and robustness
✅ Reduces beam search overconfidence
✅ Aligns better with human evaluation metrics (BLEU/ROUGE)

---

### 🔹 7. Evaluation Metrics — Measuring Text Quality

#### 🧠 BLEU (Bilingual Evaluation Understudy)

* **Used for:** Machine Translation
* **Measures:** n-gram **precision** (how many predicted n-grams appear in reference)
* **Formula:** Geometric mean of n-gram precisions × brevity penalty
* **Weakness:** Penalizes valid paraphrases, favors shorter outputs

#### 🧩 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

* **Used for:** Summarization, text generation
* **Measures:** n-gram **recall** (how much of reference text is captured)
* **Variants:**

  * ROUGE-N (n-gram recall)
  * ROUGE-L (Longest Common Subsequence)
  * ROUGE-W (Weighted)

**Comparison:**

| Metric    | Focus     | Best For      |                                      |
| :-------- | :-------- | :------------ | :----------------------------------- |
| **BLEU**  | Precision | Translation   | How accurate are model’s words?      |
| **ROUGE** | Recall    | Summarization | How much reference content captured? |

---

### 🔹 8. Key Takeaways

✅ Attention-guided decoding improves translation quality
✅ Teacher forcing stabilizes training but causes exposure bias
✅ Sampling controls creativity; beam search controls optimality
✅ MBR aligns decoding with true objective metrics
✅ BLEU & ROUGE remain standard automated evaluation metrics

---

### 🧭 Quick Mnemonics

* **“Train → Decode → Evaluate”** = core pipeline
* **Teacher forcing** = training shortcut
* **Temperature** = creativity knob
* **Beam width** = exploration scope
* **MBR** = accuracy vs similarity tradeoff
* **BLEU (Precision)**, **ROUGE (Recall)** = evaluate both sides

---

**🧩 Fine-Tuning, Representation, and Evaluation Frameworks**
*(Model adaptation, representation learning, and evaluation in modern NLP)*

---

### 🌍 Concept Overview

**Goal:** Adapt pre-trained language models (PLMs) efficiently to downstream tasks while maintaining generalization and stability.
**Focus:**

* How to represent & compare sentence meanings
* How to fine-tune large models effectively
* How to evaluate semantic understanding in a standardized way

---

### 🔹 1. Representation Learning & Semantic Similarity

Understanding how to represent text so that *semantic meaning* — not just surface form — is captured.

#### 🧠 Siamese Networks

**Concept:** Twin networks with shared weights → learn *comparable embeddings* for two inputs.
**Pipeline:**

* Input: Sentence A, Sentence B
* Both passed through the same encoder (e.g., BERT, LSTM)
* Outputs → fixed-size embeddings
* Compute **similarity score** (cosine / Euclidean)

**Used in:**

* Semantic Textual Similarity (STS)
* Sentence matching (e.g., paraphrase detection, duplicate questions)

**Advantages:**
✅ Shared weights ensure consistent representation space
✅ Works well with few-shot learning setups

---

### 🔹 2. Triplet Loss — Learning Discriminative Embeddings

**Purpose:** Encourage semantically similar sentences to be close, and dissimilar ones to be far apart in embedding space.

**Triplet:**

* **Anchor (A):** reference sentence
* **Positive (P):** semantically similar sentence
* **Negative (N):** semantically different sentence

**Loss Function:**
L = max(0, d(A, P) - d(A, N) + margin)

**Key Variants:**

* **Mean Negative:** Average distance over multiple negatives
* **Closest Negative:** Choose hardest negative (smallest distance)

**Trade-off:**

* Mean negative → stable training
* Closest negative → faster convergence but risk of instability

**Used in:**
Sentence-BERT (SBERT), text retrieval, face recognition-style embedding tasks.

---

### 🔹 3. Language Modeling Objectives for Fine-Tuning

#### 🧩 Masked Language Modeling (MLM)

* Randomly mask a portion (e.g., 15%) of input tokens
* Model predicts the masked tokens
* **Learning goal:** Deep contextual representations
* **Example:** “The cat sat on the [MASK].” → “mat”

**Why effective:**

* Enables bidirectional context (left + right)
* Core pretraining task in BERT

#### 🔸 Multi-Mask Language Modeling (MMLM)

* Variant of MLM where **multiple masks** per sequence are handled dynamically.
* Reduces overfitting to single-token predictions.
* Helps with **sentence-level understanding** (contextual reasoning).

**Insight:**
Fine-tuning on MLM or MMLM tasks can improve downstream robustness by reinforcing contextual comprehension.

---

### 🔹 4. Fine-Tuning Strategies — Efficient Adaptation of Large Models

#### ⚙️ Gradual Unfreezing

**Problem:** Fine-tuning entire pre-trained model → catastrophic forgetting.
**Solution:**

* Unfreeze layers **progressively** during training.
* Start with task-specific head → deeper encoder layers → lower embeddings.

**Benefits:**
✅ Smooth adaptation
✅ Preserves general pre-trained knowledge
✅ Common in low-resource or domain adaptation setups

---

#### 🧩 Adapter Layers

**Idea:** Add **small trainable modules** (adapters) between frozen transformer layers.

* Each adapter = bottleneck MLP (down-projection → non-linearity → up-projection)
* Freeze base model weights; train only adapters.

**Advantages:**
✅ Dramatically reduces fine-tuning cost
✅ Supports **multi-task learning** (plug different adapters per task)
✅ Faster training, less catastrophic forgetting

**Popular Implementations:**

* **AdapterFusion:** Combines multiple task adapters
* **LoRA (Low-Rank Adaptation):** Injects low-rank updates into attention weights

---

### 🔹 5. Evaluation Frameworks — Measuring Model Understanding

#### 🧠 GLUE Benchmark (General Language Understanding Evaluation)

**Purpose:** Standardized suite for evaluating **generalization** of NLP models across tasks.

**Includes 9 core tasks:**

| Task      | Type                           | Example                                   |
| :-------- | :----------------------------- | :---------------------------------------- |
| **CoLA**  | Linguistic acceptability       | “Is this sentence grammatically correct?” |
| **SST-2** | Sentiment analysis             | “positive” / “negative” classification    |
| **MRPC**  | Paraphrase detection           | “Are these two sentences equivalent?”     |
| **STS-B** | Semantic textual similarity    | Continuous similarity score               |
| **QQP**   | Quora Question Pairs           | Duplicate question identification         |
| **MNLI**  | Natural language inference     | Entailment / contradiction / neutral      |
| **QNLI**  | Question-answer entailment     | “Does the passage answer the question?”   |
| **RTE**   | Recognizing textual entailment | True / False inference                    |
| **WNLI**  | Coreference resolution         | “Who does ‘he’ refer to?”                 |

**Metric:**

* Accuracy or F1 (classification tasks)
* Pearson/Spearman correlation (similarity tasks)

**SuperGLUE:**

* Harder extension of GLUE
* Adds commonsense reasoning and multi-sentence understanding

---

### 🔹 6. Key Takeaways

✅ Siamese networks → learn comparable embeddings for semantic similarity
✅ Triplet loss → enforces distance structure in embedding space
✅ MLM/MMLM → pretraining objectives that enhance contextual learning
✅ Gradual unfreezing & adapter layers → stable, efficient fine-tuning
✅ GLUE → unified framework for evaluating linguistic and semantic competence

---

### 🧭 Quick Mnemonics

* **“Siamese twins share weights.”**
* **“Triplets teach distance.”**
* **“Mask → Predict → Understand.”**
* **“Unfreeze slowly, forget less.”**
* **“Adapters adapt, don’t overwrite.”**
* **“GLUE tests your true understanding.”**

---


