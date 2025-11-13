
---

### **1️⃣ Data Preprocessing**

**Goal:** Clean, normalize, and standardize raw text for model input.

**Key Concepts:**

* **Tokenization:** Splitting text into smaller units (words, subwords, or sentences).
* **Stopwords Removal:** Eliminating common non-informative words (e.g., *the, and, is*).
* **Stemming:** Reducing words to their root form by chopping suffixes (e.g., *playing → play*).

  * Algorithm: **Porter Stemmer**, **Snowball Stemmer**.
  * Pros: Simple, fast; Cons: May produce non-words.
* **Lemmatization:** Mapping words to their dictionary form (lemma) using linguistic rules.

  * Uses POS tagging → *better context awareness*.
  * Example: *went → go*, *better → good*.
* **Normalization:** Lowercasing, removing punctuation, expanding contractions.

**Outcome:** Clean, minimal, and uniform tokens → ready for vectorization.

---

### **2️⃣ Vector Space Models & Cosine Similarity**

**Goal:** Represent words or documents as numeric vectors for comparison and computation.

**Concepts:**

* **Bag of Words (BoW):** Counts word frequencies; ignores word order.
* **TF-IDF (Term Frequency–Inverse Document Frequency):** Weighs words based on importance across documents.

  * Formula: TF × log(N / DF).
* **Cosine Similarity:** Measures angle similarity between two vectors.

  * Formula: cos(θ) = (A · B) / (‖A‖‖B‖).
  * Range: [-1, 1] → 1 = identical direction.
* **Applications:**

  * Document similarity
  * Information retrieval
  * Search ranking

**Limitations:**

* High dimensionality, sparse vectors
* No semantic understanding (e.g., *car* and *automobile* appear unrelated)

---

### **3️⃣ Word Embedding Creation**

**Goal:** Learn dense, continuous vector representations capturing semantic meaning.

**Approaches:**

#### (a) **CBoW (Continuous Bag of Words)**

* Predict the *current word* based on surrounding *context words*.
* Example: “the ___ runs fast” → predict “dog”.
* Fast and stable for frequent words.

#### (b) **Skip-Gram**

* Predict *context words* given a *target word*.
* Works better for infrequent words.
* Example: Input = “dog” → Predicts context like {“the”, “runs”, “fast”}.

#### (c) **Negative Sampling**

* Efficient training trick for large vocabularies.
* Instead of full softmax, only updates weights for a few “negative” samples per step.

#### (d) **Word2Vec**

* Implements CBoW and Skip-Gram models.
* Learns semantic & syntactic relationships.
* Captures analogies like “king - man + woman ≈ queen”.

#### (e) **GloVe (Global Vectors)**

* Combines global co-occurrence statistics with local context windows.
* Objective: minimize difference between predicted and actual co-occurrence ratios.
* Better at global semantic consistency.

#### (f) **FastText**

* Extends Word2Vec by representing words as n-grams (subword units).
* Helps handle **out-of-vocabulary** and **morphologically rich** languages.
* Example: “playing” = {“pla”, “lay”, “ayi”, “yin”, “ing”}.

**Outcome:** Words become dense vectors in a continuous space where distance encodes meaning.

---

### **4️⃣ Self-Supervised Embedding Learning & Evaluation**

**Goal:** Automatically learn word semantics without labeled data.

**Self-Supervised Tasks:**

* **Context Prediction:** (CBoW, Skip-Gram) predict missing word/context.
* **Masked Language Modeling:** predict masked token (used in BERT).
* **Next Sentence Prediction:** learn inter-sentence coherence.

**Evaluation Methods:**

* **Intrinsic Evaluation:**

  * Word similarity tasks (e.g., cosine similarity between “king” & “queen”).
  * Analogy tasks (e.g., “Paris : France :: Tokyo : ?”).
* **Extrinsic Evaluation:**

  * Downstream NLP performance (sentiment, classification, NER).

**Modern Trends:**

* Contextual embeddings (e.g., BERT, ELMo) → depend on sentence context.
* Static embeddings (Word2Vec, GloVe) → single vector per word.

---

**Probabilistic Modeling & Language Understanding**
---

### **1️⃣ N-Gram Language Models**

**Goal:** Estimate the probability of a word sequence using preceding context.

**Core Idea:**
Language is modeled as a **probability distribution over word sequences**:
P(w₁, w₂, …, wₙ) ≈ ∏ P(wᵢ | wᵢ₋₁, …, wᵢ₋ₙ₊₁)

**Concepts & Components:**

* **Unigram Model:** P(wᵢ) — assumes independence between words.
* **Bigram Model:** P(wᵢ | wᵢ₋₁) — one preceding word as context.
* **Trigram Model:** P(wᵢ | wᵢ₋₁, wᵢ₋₂) — two-word context for better fluency.
* **Markov Assumption:** The probability of a word depends only on the last *n−1* words.
  → Simplifies computation but loses long-term dependencies.

**Smoothing Techniques:**

* **Add-One / Laplace Smoothing:** Avoids zero probabilities by adding 1 to all counts.
* **Good-Turing Discounting:** Adjusts for unseen n-grams by redistributing probabilities.
* **Kneser-Ney Smoothing:** Advanced method using continuation probabilities for better results.

**Special Tokens:**

* **<s>** → Start-of-sentence token.
* **</s>** → End-of-sentence token.
  They ensure the model learns sentence boundaries and proper context windows.

**Evaluation Metric:**

* **Perplexity (PP):** Measures how well a model predicts test data.

  * Formula: PP = 2^(− (1/N) Σ log₂ P(w₁, …, wₙ))
  * Lower perplexity ⇒ better predictive performance.

**Limitations:**

* Data sparsity (exponential growth of n-grams).
* Limited context window (short-term memory).
* Poor generalization to unseen sequences.

---

### **2️⃣ Naive Bayes for Text Classification**

**Goal:** Classify text (e.g., sentiment, topic) by computing class probabilities using Bayes’ theorem.

**Bayes’ Theorem:**
P(Class | Words) ∝ P(Words | Class) × P(Class)

**Assumptions:**

* **Conditional Independence:** All features (words) are independent given the class label.

  * Example: P(good, movie | positive) = P(good | positive) × P(movie | positive).

**Types of Naive Bayes Models:**

* **Multinomial NB:** Works with word frequencies (most common for text).
* **Bernoulli NB:** Works with binary word presence/absence.
* **Gaussian NB:** For continuous features (less common in NLP).

**Key Components:**

* **Prior Probability:** P(Class) — probability of each label (e.g., positive vs negative).
* **Likelihood:** P(Words | Class) — probability of seeing the words in a document given the class.
* **Posterior Probability:** P(Class | Words) — what we ultimately compute for classification.

**Laplacian Smoothing (Add-One):**
Prevents zero probabilities for unseen words:
P(w | c) = (Count(w, c) + 1) / (Σ Count(all_words, c) + V)
where *V* = vocabulary size.

**Applications:**

* **Sentiment Analysis:** Classify text as positive/negative/neutral.
* **Spam Filtering:** Classify emails as spam vs ham.
* **Topic Detection:** Assign text to categories.

**Strengths:**

* Simple, fast, and interpretable.
* Works surprisingly well on high-dimensional sparse text data.

**Limitations:**

* Independence assumption is unrealistic.
* Ignores word order and context relationships.

---

### **3️⃣ Parts of Speech (POS) Tagging with Markov Models**

**Goal:** Assign the most likely sequence of POS tags (e.g., Noun, Verb, Adj) to a sentence.

**Concepts:**

* Sequence labeling problem → predict *T = {t₁, t₂, …, tₙ}* for words *W = {w₁, w₂, …, wₙ}*.
* Model the joint probability:
  P(W, T) = ∏ P(wᵢ | tᵢ) × P(tᵢ | tᵢ₋₁)

**1. Markov Approximation:**
Each tag depends only on the previous tag (first-order Markov assumption).
P(tᵢ | t₁, …, tᵢ₋₁) ≈ P(tᵢ | tᵢ₋₁)

**2. Hidden Markov Model (HMM):**

* **Hidden States:** POS tags (unobserved).
* **Observations:** Words (observed).
* **Parameters:**

  * Transition Probabilities → P(tᵢ | tᵢ₋₁)
  * Emission Probabilities → P(wᵢ | tᵢ)
  * Initial Probabilities → P(t₁)

**3. Viterbi Algorithm:**

* Dynamic programming algorithm to find the most likely tag sequence.
* Steps:

  1. Initialization — Start from first word and compute probabilities.
  2. Recursion — Propagate probabilities for each subsequent tag.
  3. Termination — Select path with maximum probability.
* Output: Optimal tag sequence maximizing P(T | W).

**Example:**
Sentence: *He runs fast.*
→ Possible tags: {PRON, VERB, ADV}
→ HMM assigns probabilities and chooses most probable sequence.

**Applications:**

* Grammatical analysis
* Syntax parsing
* Preprocessing for higher-level NLP tasks

**Limitations:**

* Requires large annotated corpora for accurate transition/emission estimation.
* Struggles with long-range dependencies or ambiguous contexts.

---

**🧩 Summary Flow of Section:**
Text → Tokenized Words
↓
**N-Gram Models:** Predict next word using previous ones (language modeling)
↓
**Naive Bayes:** Classify text using probabilistic independence (text understanding)
↓
**HMM + Viterbi:** Label sequences like POS tags using transition probabilities

---

**✨ Key Takeaways:**

* Probabilistic NLP relies on **conditional probability** and **independence assumptions**.
* N-Gram models capture **short-term dependencies**; HMMs capture **sequence structure**.
* Naive Bayes provides a simple but powerful **baseline for classification tasks**.
* Together, these form the **statistical foundation** upon which modern neural NLP is built.

**🧠 Applied Probabilistic Techniques in NLP Tasks**

---

### **1️⃣ Autocorrect & Edit Distance**

**Goal:** Automatically correct spelling errors by finding the closest valid word based on minimal edit operations.

**Core Concept:**
Measure *similarity between strings* using **Edit Distance** — the minimum number of operations required to transform one word into another.

**Key Operations:**

* **Insertion:** Add a character (e.g., “aple” → “apple”)
* **Deletion:** Remove a character (e.g., “appple” → “apple”)
* **Substitution:** Replace a character (e.g., “appel” → “apple”)

**Levenshtein Distance:**

* Defines edit distance as the **minimum number of insertions, deletions, and substitutions**.
* Computed using **Dynamic Programming (DP)**.

**DP Recurrence Relation:**
If we denote D(i, j) as edit distance between prefixes word₁[0…i] and word₂[0…j]:

D(i, j) = min(
- D(i-1, j) + 1,         ← deletion
- D(i, j-1) + 1,         ← insertion
- D(i-1, j-1) + cost     ← substitution (cost = 0 if same char else 2)
- )

**Algorithmic Insight:**

* DP table of size m×n where m, n are lengths of input words.
* Time complexity: O(m × n)

**Applications:**

* **Spell Correction:** Suggest nearest word candidates based on smallest distance.
* **Fuzzy Matching:** Find similar strings in noisy datasets.
* **OCR / Speech Correction:** Identify plausible corrections for misrecognized words.

**Probabilistic Twist:**
Autocorrect can also use *Bayesian inference*:
P(correct_word | observed_word) ∝ P(observed_word | correct_word) × P(correct_word)
→ Combines likelihood of typo with prior word frequency.

---

### **2️⃣ Machine Translation using Word Vectors**

**Goal:** Translate words or phrases between languages by aligning their semantic vector spaces.

**Key Idea:**
Words with similar meanings across languages occupy **similar positions** in vector space representations.

**Steps:**

1. **Train monolingual embeddings** separately (e.g., English Word2Vec, Spanish Word2Vec).
2. **Find linear mapping (rotation matrix)** between vector spaces.

   * Optimize for minimal distance between known translation pairs:
     minimize || W × xᵢ - yᵢ ||²
     where xᵢ = source word vector, yᵢ = target word vector, W = transformation matrix.
3. **Orthogonal Mapping (Rotation Matrix):**

   * Ensures distances and relationships are preserved.
   * Solved via **Procrustes alignment** using Singular Value Decomposition (SVD).

**Mathematical Formulation:**
W = UVᵀ, where XᵀY = USVᵀ (from SVD decomposition).

**Applications:**

* **Bilingual dictionary induction:** Automatically map words across languages.
* **Cross-lingual embeddings:** Represent multiple languages in a shared semantic space.
* **Low-resource translation:** Use transfer learning from high-resource language embeddings.

**Example:**
“King” in English maps near “Rey” in Spanish after vector alignment.

**Benefits:**

* Simple, interpretable, and data-efficient.
* Enables unsupervised or weakly supervised translation.

---

### **3️⃣ Nearest Neighbors & Approximate Similarity Search**

**Goal:** Efficiently find the most similar words, sentences, or documents in high-dimensional vector space.

**Core Idea:**
Similarity is often defined via **distance metrics** (cosine, Euclidean).
Finding neighbors in large datasets is computationally expensive — hence approximate methods.

**Techniques:**

**A. Nearest Neighbors (NN):**

* **Brute-force search:** Compute distance from query to all vectors.
* **Drawback:** O(N) per query — infeasible for millions of embeddings.

**B. Locality Sensitive Hashing (LSH):**

* Hash similar items into the same “bucket” with high probability.
* Uses random projections to reduce dimensionality while preserving similarity.
* Approximate similarity search: trade small accuracy loss for massive speed gain.

**LSH Pipeline:**

1. Generate random projection hyperplanes.
2. Compute hash for each vector (sign of dot product).
3. Compare hashes — fewer distance computations needed.

**Applications:**

* Semantic search (find similar meanings).
* Duplicate detection.
* Large-scale recommendation systems.

**C. Approximate Nearest Neighbors (ANN):**

* Frameworks like **FAISS**, **Annoy**, **ScaNN** efficiently retrieve top-k similar vectors.
* Use tree-based or graph-based indices (e.g., HNSW).
* O(log N) retrieval instead of O(N).

**Probabilistic Perspective:**
Approximation relies on probabilistic guarantees that *similar items hash together with higher probability* than dissimilar ones.

---

**🧩 Summary Flow of Section:**
Text Input
↓
**Autocorrect:** Correct noisy text using edit distances and probabilistic priors
↓
**Translation via Embedding Alignment:** Map semantic spaces across languages using vector transformations
↓
**Similarity Search:** Retrieve or rank similar items using probabilistic hashing and approximate inference

---

**✨ Key Takeaways:**

* Probabilistic reasoning supports **robust real-world NLP applications** under uncertainty.
* **Edit distance** quantifies textual similarity — foundation for correction tasks.
* **Vector alignment** bridges languages by leveraging geometric and probabilistic principles.
* **LSH & ANN** scale similarity-based NLP tasks to massive datasets efficiently.

These methods form the **practical bridge between theory and deployment**, connecting probabilistic modeling to real-world NLP systems like autocorrect, translation, and search engines.
