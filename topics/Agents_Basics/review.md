
---

# 📘 **Fundamentals of AI Agents**

---

## ## 1. **Think → Act → Observe (TAO) Cycle**

At the core of every agentic system is a repeatable decision loop:

### **1. Think**

* The agent interprets the current context, instructions, memory, and prior observations.
* Internal reasoning (chain-of-thought, planning, goal decomposition).
* Often implemented through hidden reasoning tokens not exposed to the user.

### **2. Act**

* The agent selects and executes an action:

  * Function/tool call
  * External API use
  * Code execution
  * Database retrieval
  * Query/search

### **3. Observe**

* The agent receives feedback/state changes from the environment:

  * Tool output
  * Errors
  * New data
  * User feedback
* This observation updates the internal state and restarts the loop.

### **Why TAO Matters**

* Converts static LLMs into dynamic problem-solvers.
* Enables multi-step reasoning.
* Makes models adaptable to new information over time.

---

# ## 2. **ReAct Framework**

**ReAct (Reason + Act)** enables LLMs to:

* Generate **explicit reasoning traces**.
* Take actions based on those traces.
* Continue iterating until the task is solved.

### **Key Components**

* **Thought** → model explains intermediate reasoning.
* **Action** → model decides a tool call or query.
* **Observation** → tool/environment returns data.
* **Answer** → final output delivered to user.

### **Benefits**

* Improved interpretability.
* Easier debugging because reasoning is structured.
* Supports multi-step tool usage and dynamic planning.

---

# ## 3. **Three Kinds of Action Agents**

### ### 3.1 **JSON Agents**

* Output structured JSON containing instructions.
* Acts as a high-level orchestrator.
* Good for deterministic pipelines.
* Limitation: cannot execute real logic or handle complex tasks alone.

---

### ### 3.2 **Function Calling Agents**

* Use LLM-native function-calling API.
* Model selects:

  * Which function to call.
  * What arguments to pass.
* Very reliable and restricted — great for enterprise systems.

**Strengths**

* Strong schema validation
* Perfect for workflows, APIs, retrieval, structured tools
* Prevents hallucinated commands

**Limitations**

* No dynamic code generation
* Limited expressiveness compared to agents that can write and run code

---

### ### 3.3 **Code Agents**

* Generate and execute **Python/JS code** dynamically in a secure sandbox.
* Extremely powerful for:

  * Data analysis
  * Automation
  * Multi-step computation
  * Building tools on the fly
  * Simulations
  * Complex math and algorithmic tasks

---

### ⭐ **Advantages of Code Agents**

1. **Expressiveness** → can solve *any* computable problem.
2. **Self-debugging** → agent can correct its own code via feedback.
3. **Scalable** → can build new tools during runtime.
4. **Highly flexible** → not limited by predefined functions.
5. **Strong reliability** → exact reproducible computation.

---

# ## 4. **Observations & Adaptive Strategy**

### ### Types of Observations

* **Tool Output** (API responses, DB results)
* **State Changes** (variables, memory updates)
* **Errors** (runtime exceptions, missing keys)
* **Environment feedback** (web search results, sensor data)
* **User feedback** (explicit or implicit)

### ### How Agents Use Observations

* **Collect Feedback** → evaluate last action success/failure.
* **Append Results** → include results in an internal working memory.
* **Adapt Strategy** → re-plan based on new info.

### **Adaptive Strategy Example**

* If retrieval returns irrelevant results → rewrite query.
* If code execution errors → fix code and retry.
* If user is getting impatient → simplify answers.

---

# ## 5. **Agentic RAG (Retrieval-Augmented Generation)**

Agentic RAG enhances traditional RAG with multi-step agent reasoning.

### **Components**

### ### 5.1 **Query Reformation**

* Agent rewrites the query to improve retrieval.
* Techniques: expansion, simplification, disambiguation.

### ### 5.2 **Multi-step Retrieval**

* Agent retrieves *iteratively*:
  Step 1: metadata → Step 2: specific docs → Step 3: data fusion.

### ### 5.3 **Source Integration**

* Agent merges info from:

  * Vector DB
  * SQL databases
  * APIs
  * File systems
  * Web scraping
  * LLM tools

### ### 5.4 **Result Validation**

* Check factual consistency.
* Compare across multiple sources.
* Chain-of-thought to justify answers.
* Reject low-confidence or hallucinated content.

---

# ## 6. **Benefits of Multi-Agent Systems**

### **Why use multiple agents instead of one?**

1. **Specialization**

   * One agent for data retrieval, another for code, another for reasoning.

2. **Modularity**

   * Easier debugging and maintenance.

3. **Parallelism**

   * Agents run in parallel → faster pipelines.

4. **Reduced hallucinations**

   * Reviewer agents verify output.

5. **Complexity Handling**

   * Decompose big tasks into smaller sub-agents.

6. **Human-AI collaboration**

   * Human approval loops between agents.

---

# ## 7. **LlamaIndex Components**

LlamaIndex provides an agentic framework built around retrieval and tools.

---

### ### 7.1 **Key Components**

* **Nodes/Data** → smallest unit of knowledge
* **Indexes** → vector stores, keyword tables, graph indexes
* **Retrievers** → query specific pieces of context
* **Query Engines** → orchestrate retrieval + synthesis
* **Agents** → tool-calling, reasoning entities
* **Tools** → APIs, code tools, DB connectors

---

### ### 7.2 **Agent Types**

* **Function calling agents**
* **ReAct agents**
* **Query-engine-enabled agents**
* **Multi-agent systems with routers**

---

### ### 7.3 **Tools in LlamaIndex**

#### **Function Tools**

* Standard function-calling with schema.

#### **Query Engine Tools**

* A retriever + synthesizer wrapped as a tool.
* Great for multi-database or multi-index RAG.

#### **Tool Specs**

* Defines the exact interface, input schema, output schema.

#### **Utility Tools**

* Web search
* File reading/writing
* Calculator
* Code execution

---

# ## 8. **Maintaining State**

Agents often need persistent context.

### **Ways to Maintain State**

* Short-term memory (trace of actions + observations)
* Long-term vector memory
* State storage (DB, Redis, JSON logs)
* Session-aware prompts: user preferences, conversation history

### **Why State Matters**

* Enables multi-step workflows
* Supports human-in-the-loop systems
* Allows recovery from errors
* Powers personalization

---

# ## 9. **Streaming Key Steps**

Agents can stream:

* Partial reasoning
* Intermediate tool results
* Progress notifications
* Final responses

Benefits:

* Transparency
* Better UX
* Human monitoring during long tasks

---

# ## 10. **Human in the Loop (HITL)**

Used in:

* High-risk decisions
* Multi-agent approvals
* Route selection
* Query rewriting confirmation
* Execution of potentially unsafe operations

Forms of HITL:

* Pre-action approval
* Post-action validation
* Override or correction
* Feedback loops to improve future performance

---

# ## 11. **Agentic Workflows vs Autonomous Agents**

| Aspect      | Agentic Workflow                    | Autonomous Agent                        |
| ----------- | ----------------------------------- | --------------------------------------- |
| Control     | Human-defined pipeline              | Agent plans actions itself              |
| Determinism | High                                | Medium/low                              |
| Flexibility | Lower                               | Very high                               |
| Safety      | Higher                              | Requires constraints                    |
| Use Cases   | Enterprise apps, production systems | Creative tasks, exploration, simulation |

**Conclusion:**

* Workflows = predictability
* Autonomous agents = adaptability

---

# ## 12. **LangGraph**

LangGraph = framework for building **stateful, multi-step agent workflows**.

### **Core Features**

* Node-based DAG structure
* State management
* Memory built-in
* Supports cycles (looping)
* Agent-to-agent communication
* Persistence and observability

Perfect for:

* Agent orchestration
* Multi-agent RAG
* Code execution agents
* Complex reasoning pipelines

---

# ## 13. **Agent Observability & Evaluation**

Agent systems need monitoring similar to production ML systems.

---

### ### 13.1 **Key Metrics**

#### **Accuracy**

* Whether tasks are solved correctly.

#### **Problem Solving Ability**

* Multi-step reasoning correctness.

#### **Information Retrieval Quality**

* Recall, precision of retrieved context.

#### **User Satisfaction**

* Explicit ratings, conversational quality.

#### **User Feedback**

* Thumbs up/down
* Suggested changes
* Correction loops

#### **Implicit Feedback**

* User re-queries
* Hesitation time
* Scroll depth
* Interaction patterns

#### **Latency**

* Time per action/response
* Tool call delays

#### **Token Usage**

* Input/output tokens
* Cost optimization

#### **Request Errors**

* API failures
* Tool errors
* Code execution exceptions

#### **Automated Evaluation**

* Synthetic benchmarks
* Gold-standard prompts
* Rubric-based scoring

---

# ## 14. **Function Calling by Fine-Tuning the Model**

Modern LLMs can be fine-tuned to **natively perform function-calling** with:

* Lower latency
* Higher reliability
* Less prompt engineering

---

### ### 14.1 **Supervised Fine-Tuning (SFT)**

* Train the model on input → tool-call-output pairs.
* Used in:

  * Structured agents
  * RAG planners
  * Enterprise workflows

---

### ### 14.2 **LoRA Fine-Tuning**

* Parameter-efficient method
* Injects low-rank matrices into model layers
* Ideal for:

  * Small datasets
  * Cost-efficient tuning
  * Local deployments

---

# 🎯 **Final Takeaway — What You Should Remember**

* Agents operate via **Think–Act–Observe** loops.
* **ReAct** enables reasoning + dynamic tool usage.
* Code agents are the most flexible and powerful.
* Observations enable adaptation and strategy refinement.
* Agentic RAG is the next frontier of retrieval systems.
* Multi-agent systems boost reliability, specialization, and scalability.
* LlamaIndex + LangGraph are key frameworks for building agent ecosystems.
* State, streaming, HITL, and workflow orchestration form the backbone of real agent systems.
* Evaluation of agents is multi-dimensional and critical for production readiness.
* Fine-tuning (SFT, LoRA) enhances tool-calling and agent behavior.

---