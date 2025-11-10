🧩 **Foundations of Sequential Modeling — Mind-Map Cheat Sheet**

---

### 🧠 **Core Idea**

Sequential modeling enables neural networks to **process temporal or ordered data** — like text, speech, or time series — where **context and sequence order matter**.

---

### 🔹 **1. Recurrent Neural Networks (RNN)**

* **Purpose:** Handle sequential data by maintaining a **hidden state** that carries information through time.
* **Architecture:**

  * Input $$(  x_t  )$$ → hidden state $$(  h_t = f(W_h h_{t-1} + W_x x_t + b)  )$$
  * Output $$(  y_t = g(W_y h_t + c)  )$$
* **Key Property:** Shares weights across time steps → enables temporal generalization.
* **Activation functions:** Commonly **tanh** or **ReLU**.
* **Problem:** Struggles with long-term dependencies due to vanishing/exploding gradients.

---

### ⚠️ **2. Vanishing/Exploding Gradient Problem**

* **Cause:** During backpropagation through many time steps, gradients **shrink (vanish)** or **grow (explode)** exponentially.
* **Consequence:** Early time steps have little to no influence on the final prediction.
* **Symptoms:**

  * Slow or no learning for long sequences
  * Poor gradient flow → unstable training
* **Solutions:**

  * Gradient clipping (for explosion)
  * Gated architectures (for vanishing) → GRU, LSTM

---

### 🔒 **3. Gated Architectures — GRU & LSTM**

#### 🧩 **LSTM (Long Short-Term Memory)**

* **Goal:** Retain long-term information by controlling what to remember/forget.
* **Components:**

  * **Forget gate:** $$(  f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)  )$$
  * **Input gate:** $$(  i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)  )$$
  * **Candidate cell:** $$(  \tilde{C_t} = \tanh(W_c [h_{t-1}, x_t] + b_c)  )$$
  * **Cell state update:** $$(  C_t = f_t * C_{t-1} + i_t * \tilde{C_t}  )$$
  * **Output gate:** $$(  o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)  )$$
  * **Hidden state:** $$(  h_t = o_t * \tanh(C_t)  )$$
* **Advantage:** Effective for long-term dependencies; avoids vanishing gradient.

#### ⚙️ **GRU (Gated Recurrent Unit)**

* **Simplified version of LSTM** (fewer gates, no separate cell state).
* **Gates:**

  * **Update gate:** $$(  z_t = \sigma(W_z [h_{t-1}, x_t])  )$$
  * **Reset gate:** $$(  r_t = \sigma(W_r [h_{t-1}, x_t])  )$$
  * **Candidate state:** $$(  \tilde{h_t} = \tanh(W_h [r_t * h_{t-1}, x_t])  )$$
  * **Hidden state:** $$(  h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h_t}  )$$
* **Pros:**

  * Faster training
  * Fewer parameters
  * Performs comparably to LSTM on many tasks

**Summary:**

| Feature    | LSTM                      | GRU               |
| ---------- | ------------------------- | ----------------- |
| Gates      | 3 (input, forget, output) | 2 (update, reset) |
| Cell state | Yes                       | No                |
| Complexity | Higher                    | Lower             |
| Speed      | Slower                    | Faster            |

---

### 🔁 **4. Bidirectional RNNs**

* **Idea:** Use two RNNs — one reads the sequence **forward**, the other **backward**.
* **Output:** Combines both directions’ hidden states → richer context.
* **Equation:** $$(  h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]  )$$
* **Benefit:** Captures both **past and future** context for each token (useful in NLP tasks like POS tagging or sentiment analysis).
* **Limitation:** Not suitable for real-time or streaming tasks (requires full sequence).

---

### 🧩 **Summary Mind-Flow**

**Sequential Modeling**
⬇
**RNN:** Basic memory across time
⬇
**Vanishing Gradients:** Challenge in long sequences
⬇
**GRU & LSTM:** Solve gradient issues using gates
⬇
**Bidirectional RNN:** Add backward context for richer understanding

---

💬 **Word Representations & Embeddings — Mind-Map Cheat Sheet**

---

### 🧠 **Core Idea**

Computers cannot process raw text directly — we convert words into **dense numerical vectors (embeddings)** that capture **semantic meaning and contextual similarity**.
Goal: Represent words such that **similar words have similar vector representations**.

---

### 🔹 **1. Word Embeddings & Embedding Matrix**

* **Definition:**
  A **word embedding** is a dense vector representation (e.g., 100–300 dimensions) that encodes semantic properties of a word.

* **Embedding Matrix:**

  * A learnable parameter matrix $$(  E \in \mathbb{R}^{V \times d}  )$$, where
    ( V ) = vocabulary size, ( d ) = embedding dimension.
  * Each word index maps to a row vector (embedding).
  * Used as input layer in neural NLP models.

* **Advantages of dense embeddings:**
  ✅ Capture semantics (e.g., king - man + woman ≈ queen)
  ✅ Reduce dimensionality vs. one-hot vectors
  ✅ Enable transfer learning from pre-trained models

---

### 🧩 **2. Word2Vec — Predictive Embeddings**

**Idea:** Learn embeddings based on **context prediction**.
Two main architectures by Mikolov et al. (Google, 2013):

#### ⚙️ a. CBOW (Continuous Bag of Words)

* Predict **target word** from **context words**.
* Example: “The ___ barks loudly” → predict *dog*.
* Input: average of context embeddings → Output: target word probability.

#### ⚙️ b. Skip-Gram

* Predict **context words** given the **target word**.
* Example: Input = *dog* → predict words like *barks, pet, animal*.
* Works better with small data and rare words.

---

### 🔸 **3. Negative Sampling**

* **Purpose:** Efficiently train Word2Vec models by avoiding large softmax computations over the entire vocabulary.
* **Approach:**

  * For each positive (target, context) pair, sample a few **negative examples** (random unrelated words).
  * Optimize using logistic regression (1 for real pair, 0 for fake).
* **Benefit:** Greatly speeds up training while maintaining accuracy.

---

### 🧮 **4. GloVe (Global Vectors for Word Representation)**

* **Developed by:** Stanford NLP Group.
* **Concept:** Combines **global co-occurrence statistics** with local context learning.
* **Objective:**
  Learn embeddings such that **dot product of two word vectors approximates log of their co-occurrence probability**.
  $$w_i^T w_j + b_i + b_j \approx \log( X_{ij} )$$
  where $$(  X_{ij}  )$$ = number of times word *j* appears in context of *i*.
* **Difference from Word2Vec:**

  * Word2Vec → local context window
  * GloVe → global co-occurrence matrix
* **Result:** Captures both semantic and syntactic relationships.

---

### ❤️ **5. Sentiment Classification using Word Embeddings (with RNN)**

* **Pipeline:**

  1. Tokenize sentences → convert to word indices.
  2. Use **embedding layer** (pre-trained or trainable).
  3. Feed sequence of embeddings into **RNN / LSTM**.
  4. Final hidden state → fully connected layer → sentiment label.
* **Advantage:**

  * Embeddings capture meaning beyond individual tokens.
  * RNN captures context and sentiment flow across words.

---

### ⚖️ **6. Reducing Bias in Word Embeddings**

* **Problem:**
  Word embeddings learn from text corpora, which often reflect **societal biases**.
  Example: “doctor – man + woman ≈ nurse”.
* **Goal:** Debias word representations.
* **Common Techniques:**

  * **Identify bias direction:** e.g., gender direction from pairs (he–she, man–woman).
  * **Neutralize:** Remove bias component from neutral words (e.g., “doctor”).
  * **Equalize:** Make gendered pairs equidistant (e.g., “king” ↔ “queen”).
* **Post-processing approach (Bolukbasi et al., 2016):** Linear algebraic debiasing.
* **Modern trends:** Use contextual embeddings (BERT, GPT) where bias can be mitigated with fine-tuning.

---

### 🧩 **Summary Mind-Flow**

**Text Data**
⬇
**Word Embeddings** — map words to vectors
⬇
**Word2Vec** — local predictive model (CBOW / Skip-Gram)
⬇
**Negative Sampling** — efficient training
⬇
**GloVe** — global co-occurrence model
⬇
**Applications:** Sentiment classification using RNN
⬇
**Ethical Aspect:** Debiasing word representations

---
🔁 **Sequence Modeling for NLP Tasks — Mind-Map Cheat Sheet**

---

### 🧠 **Core Idea**

Sequence-to-sequence (Seq2Seq) architectures enable models to **convert one sequence into another**, e.g., translating a sentence from English → French or summarizing a paragraph.
They combine **encoder–decoder** structures and **probabilistic decoding** strategies.

---

### 🔹 **1. Sequence-to-Sequence (Seq2Seq) Architecture**

#### 🧩 **Concept**

* Composed of two RNNs (or variants like LSTM/GRU):

  * **Encoder:** Reads input sequence and compresses it into a **context vector** $$(  h_T  )$$.
  * **Decoder:** Generates output sequence using that context as initial state.

#### ⚙️ **Flow**

Input: “I love NLP” → Output: “J’adore le NLP”

1. Encoder processes words sequentially → produces hidden states $$(  h_1, h_2, ..., h_T  )$$.
2. Final state $$(  h_T  )$$ = compressed summary of input.
3. Decoder generates words step by step using previous output + hidden state.

#### 📘 **Equations**

Encoder:
$$h_t = f( W_h h_{t-1} + W_x x_t )$$
Decoder:
$$s_t = f( W_s s_{t-1} + W_y y_{t-1} + W_c h_T )$$

#### ✅ **Advantages**

* Works well for variable-length input/output.
* Effective for translation, summarization, Q&A, etc.

#### ⚠️ **Limitations**

* Fixed-length context vector → **information bottleneck** for long sentences.
* Hard for encoder to capture all input details → leads to degraded performance on long sequences.
  (*Attention mechanism later solves this!*)

---

### 🌍 **2. Neural Machine Translation (NMT)**

* **Definition:**
  Task of automatically translating a sentence from one language to another using neural networks.
* **Earlier Approaches:** Statistical phrase-based models (SMT).
* **Modern Approach:** Seq2Seq + Attention.

#### 🧠 **Key Concepts**

* Encoder captures source language meaning.
* Decoder generates target language sentence word-by-word.
* Trained end-to-end on large bilingual corpora.

#### 🏗️ **Architecture Variants**

* **Basic RNN Encoder–Decoder (Cho et al., 2014)**
* **LSTM-based NMT (Sutskever et al., 2014)**
* **Attention-based NMT (Bahdanau et al., 2015)** — improved performance for long sentences.

#### ⚙️ **Loss Function**

Cross-entropy between predicted and actual next word.
$$\mathcal{L} = -\sum_{t} \log P( y_t | y_{<t}, x )$$

---

### 🧮 **3. Decoding Strategies**

Once the model learns to predict probabilities for next words, we need a **search strategy** to choose the best possible output sequence.

#### ⚡ **a. Greedy Search**

* At each time step, pick the **highest probability word**.
* Fast and simple, but **locally optimal** → can miss better global sequences.
* Example:

  * Model predicts: [“I”, “am”, “fine”] (prob 0.6 each step)
  * But [“I”, “am”, “okay”] might have higher total probability (0.7 overall).

#### 💡 **b. Beam Search**

* Keeps **top-k sequences** (beam width) at each step instead of just one.
* Expands each candidate → keeps only top scoring ones.
* Better global optimization than greedy.
* **Beam width (k):** Controls trade-off between accuracy and speed.

#### ⚠️ **Problems with Beam Search**

* **Length bias:** Prefers shorter sentences since probabilities multiply (more words → smaller total prob).
* **Diversity issue:** Beams may converge to similar outputs.

#### 🧭 **Solutions**

* **Normalized log-likelihood objective:**
  Divide cumulative log-probability by sequence length to balance short/long output bias.
  $$\text{Score} = \frac{1}{T} \sum_t \log P( y_t | y_{<t}, x )$$
* **Diverse beam search:** Introduce diversity-promoting penalties.

---

### 🧩 **4. Putting It All Together**

**Input Sequence → Encoder → Context Vector → Decoder → Output Sequence**

Example:

> “Where is the library?” → “¿Dónde está la biblioteca?”

🧠 Training Objective:
Maximize likelihood of target sequence given input.

🧮 Decoding Objective:
Find sequence with highest normalized probability using beam search.

---

### 🧩 **Summary Mind-Flow**

**Sequential Data → Seq2Seq Architecture**
⬇
**Encoder–Decoder:** Encodes input → decodes output
⬇
**Application:** Neural Machine Translation
⬇
**Inference:** Greedy vs. Beam Search
⬇
**Optimization:** Normalized log-likelihood for better sequence scoring

---
⚡ **Attention Mechanisms & the Transformer Revolution — Mind-Map Cheat Sheet**

---

### 🧠 **Core Idea**

Traditional Seq2Seq models compress the entire input sequence into a **single context vector**, causing an **information bottleneck**.
**Attention mechanisms** solve this by allowing the model to **focus selectively** on relevant parts of the input at each decoding step — the foundation of **Transformers**, which now dominate modern NLP.

---

### 🔹 **1. Attention Mechanism — The Breakthrough**

#### 💡 **Concept**

Instead of relying only on the encoder’s final hidden state, the decoder looks at **all encoder states** and computes a **weighted average** (context vector) based on how relevant each input token is.

#### ⚙️ **Computation Steps**

For each output time step ( t ):

1. **Score alignment** between current decoder state $$(  s_t  )$$ and each encoder state $$(  h_i  )$$:
   $$e_{ti} = \text{score}( s_t, h_i )$$

2. **Convert scores to attention weights (softmax):**
   $$\alpha_{ti} = \frac{\exp( e_{ti} )}{\sum_j \exp( e_{tj} )}$$

3. **Context vector:**
   $$c_t = \sum_i \alpha_{ti} h_i$$

4. **Combine with decoder state:**
   $$s'_t = f( s_t, c_t )$$

#### 📘 **Common Scoring Functions**

* Dot product: $$(  e_{ti} = s_t^T h_i  )$$
* Additive (Bahdanau attention): $$(  e_{ti} = v_a^T \tanh(W_a [s_t; h_i])  )$$

#### ✅ **Benefits**

* Removes fixed-length bottleneck.
* Improves translation quality for long sentences.
* Adds **interpretability** (which words are “attended” to).

---

### 🔸 **2. Self-Attention — The Heart of Transformers**

#### 🧩 **Concept**

While attention in Seq2Seq links **encoder ↔ decoder**,
**self-attention** links **tokens within the same sequence**, letting each token **attend to all others**.

This enables learning **contextual relationships** without recurrence.

#### ⚙️ **Mechanism**

Each token produces three vectors:

* **Query (Q)** — what this token is looking for.
* **Key (K)** — what information this token offers.
* **Value (V)** — actual information content.

The self-attention output is a **weighted sum of values**, where weights come from **similarity between Q and K**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}}\right )V$$

* $$(  d_k  )$$: dimensionality of the key vectors (used for scaling stability).

#### 🧠 **Intuition**

Each token decides **how much attention to pay** to every other token.
Example: In “The cat sat on the mat,”
“sat” attends strongly to “cat” (subject) and “mat” (object).

---

### 🔁 **3. Multi-Head Attention**

#### ⚙️ **Purpose**

Instead of a single attention operation, perform **multiple attention heads in parallel** to capture different relational patterns (syntax, semantics, etc.).

Each head has its own Q, K, V projections:

$$\text{head}_i = \text{Attention}( QW_i^Q, KW_i^K, VW_i^V )$$
Then combine:

$$\text{MultiHead}(Q, K, V) = \text{Concat}( \text{head}_1, ..., \text{head}_h )W^O$$


#### ✅ **Advantages**

* Allows learning **multiple perspectives** of relationships.
* Captures richer dependencies between words.
* Improves generalization and performance.

---

### 📍 **4. Positional Encoding**

Since Transformers **lack recurrence**, they need a way to represent **word order**.

#### 🧮 **Solution:** Add **positional encodings** to word embeddings before feeding into the Transformer.

**Formula:**

$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{model}}}\right )$$


$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{model}}}\right )$$

#### 💡 **Intuition**

* Sine and cosine functions provide a **unique positional signature** for each token.
* The model learns relative positions and distances implicitly.

---

### 🏗️ **5. Transformer Architecture**

#### 🔺 **Structure**

* **Encoder:** Stack of identical layers (Self-Attention + Feed-Forward + LayerNorm).
* **Decoder:** Similar, but adds cross-attention to encoder outputs.
* **No RNNs or CNNs!**

#### ⚙️ **Encoder Flow**

Input embeddings + positional encodings
→ Multi-head self-attention
→ Feed-forward network
→ Residual connection + Layer normalization

#### ⚙️ **Decoder Flow**

Target embeddings + positional encodings
→ Masked self-attention (prevents peeking ahead)
→ Encoder–decoder attention
→ Feed-forward + normalization
→ Output softmax over vocabulary

#### 🔄 **Training**

Objective: Minimize cross-entropy loss over target sequence predictions.
Optimization: Adam with warmup schedule (learning rate gradually increases then decays).

---

### 🚀 **6. Transformer Advantages**

✅ Entire sequence processed **in parallel** (no recurrence).
✅ Better long-range dependency handling.
✅ Scales efficiently to massive datasets.
✅ Forms the foundation for models like **BERT, GPT, T5, and LLaMA**.

---

### 🧩 **7. Evolution Timeline**

RNN → LSTM → Attention → Self-Attention → Transformer → Pre-trained LMs (BERT, GPT)

---

### 🧩 **Summary Mind-Flow**

**Information Bottleneck in Seq2Seq**
⬇
**Attention Mechanism** — dynamic focus on relevant encoder states
⬇
**Self-Attention** — relate all tokens to each other
⬇
**Multi-Head Attention** — multiple relationship perspectives
⬇
**Positional Encoding** — inject sequence order
⬇
**Transformer Architecture** — parallelized, scalable model
⬇
**Result:** Foundation for modern NLP (BERT, GPT, etc.)

---