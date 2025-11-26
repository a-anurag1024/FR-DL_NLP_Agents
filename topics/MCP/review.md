
---

# **📘 Model Context Protocol (MCP) **

---

# **1. What is MCP?**

**Model Context Protocol (MCP)** is an open protocol designed to standardize how AI models (LLMs) interact with external systems—tools, APIs, databases, files, or custom logic.

### **Core Purpose**

* Provide **safe**, structured access to external capabilities.
* Avoid model hallucinations by exposing the “world” to the model cleanly.
* Ensure **interoperability** across tools, hosts, frameworks, and models.

### **Key Value Proposition**

* Decouples **models**, **tools**, and **clients**.
* Standard way to integrate **AI agents**, **apps**, and **dev tools**.
* Works locally and cloud-native.
* Extends classic tool-calling with **streaming**, **resource browsing**, **prompts**, and **sampling**.

---

# **2. The *m × n Integration Problem***

Traditionally:

* Each model vendor integrates separately with each tool.
* Leads to **m models × n tools** → m·n custom integrations.

**MCP solves this:**

* Tools implement MCP once.
* Hosts & models integrate MCP once.
* Now: **m + n** integrations instead of m·n.

---

# **3. MCP Terminologies**

## **Host**

* Environment that runs the **LLM**.
* Initiates MCP connections to servers.
* Examples:

  * ChatGPT
  * Claude Desktop
  * IDEs running Continue
  * Browser tools like Codeium

**Host = Client of model + Client to MCP servers**

---

## **Client**

* The **entity that establishes connection** to an MCP server.
* Could be the host itself or a UI wrapper using MCP client SDK.

**Client = the one calling the server**

---

## **Server**

* The external system that exposes:

  * **Tools**
  * **Resources**
  * **Prompts**
  * **Sampling interfaces**

Servers can be:

* Local scripts (Python, JS)
* APIs wrapped via Gradio
* Database connectors
* File systems exposed via MCP

**Server = exposes capabilities**

---

# **4. MCP Capabilities**

### **(1) Tools**

* Functions exposed to the model.
* Similar to “function calling”.
* Defined with:

  * `name`
  * `description`
  * `input_schema` (JSON Schema)

---

### **(2) Resources**

* Objects that can be:

  * Listed
  * Read
  * Watched (stream updates)
* Examples:

  * Files
  * DB tables
  * API endpoints
  * Logs

---

### **(3) Prompts**

Server can expose structured prompts, e.g.,

* Templates
* System prompts
* Reusable instructions

Useful to enforce consistency + control hallucinations.

---

### **(4) Sampling**

Allows tools/servers to ask the **model to generate text** inside a tool call.
Useful for:

* Recursive reasoning
* Tool-initiated model queries
* Co-agent workflows

---

# **5. Communication Protocol**

MCP uses **JSON-RPC 2.0**.

## **Three message types**

### **(A) Request**

* From client → server
* Requires a response
* Example:

```json
{"jsonrpc":"2.0","id":1,"method":"listTools"}
```

---

### **(B) Response**

* From server → client
* Contains result or error

```json
{"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

---

### **(C) Notification**

* No response expected
* Used for events such as:

  * Resource updates
  * Logging
  * Progress streaming

```json
{"jsonrpc":"2.0","method":"resourcesUpdated","params":{...}}
```

---

# **6. Transport Mechanisms**

MCP is transport-agnostic. Two major modes:

## **1. STDIO (default for local)**

* Server runs as a subprocess.
* Communication over stdin/stdout.
* Very fast and secure.

**Used by**: Claude Desktop, Continue, local MCP tooling.

---

## **2. HTTP (Cloud or Remote)**

Two subtypes:

### **HTTP + SSE**

* JSON-RPC over HTTP
* Streaming via Server-Sent Events

### **Streamable HTTP**

* Bidirectional streaming
* For long-running or interactive tools

**Used for remote/cloud MCP servers.**

---

# **7. Interaction Lifecycle**

### **1. Initialization**

* Host connects to server (via stdio or HTTP)
* Exchange: capabilities, version info, supported features.

---

### **2. Discovery**

* Host asks server:

  * `listTools`
  * `listResources`
  * `listPrompts`
* Model/host understands what can be done.

---

### **3. Execution**

* Model uses tools via Requests.
* Server executes logic.
* Streams intermediate messages (notifications).
* Server returns final response.

---

### **4. Termination**

* Graceful shutdown.
* Used when:

  * User session ends.
  * Host disconnects.
  * Server process exits.

---

# **8. MCP SDK from Anthropic**

Anthropic provides official SDKs:

## **Languages**

* **TypeScript**
* **Python**

## **What SDK Provides**

* RPC message helpers
* Tool declaration utilities
* Resource provider classes
* Prompt registry
* Sampling interfaces (TS only for now)
* Automatic streaming management

## **Typical Server Structure:**

### Python Server Example Structure

```python
from mcp.server import Server

server = Server("my-mcp-server")

@server.tool()
def search(query: str):
    ...
```

### TS Server Example Structure

```ts
const server = new Server({
  name: "mcp-ts-server"
});

server.tool("search", { ...schema... }, async (args) => {...});
```

---

# **9. MCP Clients & Implementations**

## **A. UI-Based Hosts**

These embed MCP clients internally.

### **ChatGPT**

* Runs MCP servers in the background.
* Allows connecting local tools.

### **Claude Desktop**

* Core reference host for MCP.
* Uses stdio-based local execution.

### **Continue (VSCode / JetBrains)**

* Acts as an MCP host.
* Lets IDE tools integrate with LLMs.

---

## **B. Gradio as MCP Server**

* Gradio can wrap existing ML or data pipelines.
* Expose them as **tools/resources**.
* Useful for:

  * Model inference endpoints
  * Local applications
  * Rapid MCP server prototyping

Frameworks for this:

* `gradio-mcp` integrations
* Custom Python MCP server using Gradio IO

---

## **C. Gradio as MCP Client (via Smolagents)**

* **Smolagents** provides MCP client implementation.
* So Gradio apps can *consume* external MCP servers.
* Allows creation of UI agents powered by:

  * Local file servers
  * DB servers
  * Code execution servers

---

## **D. Local implementations**

### **Using Continue**

* Local dev tools
* Code search
* Knowledge base integration

### **Using Claude Desktop**

* File access
* System tool wrappers

### **Using Python Scripts**

* Write your own MCP server for:

  * Filesystem access
  * API wrappers
  * DB connectors

---

# **10. Summary Table**

| Topic                  | Summary                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| **What is MCP**        | Open protocol to connect LLMs with external systems.                     |
| **m×n Problem**        | MCP reduces integrations from m·n to m+n.                                |
| **Host/Client/Server** | Host = LLM env; Client = connects; Server = exposes capabilities.        |
| **Capabilities**       | Tools, Resources, Prompts, Sampling.                                     |
| **Protocol**           | JSON-RPC 2.0 with Request/Response/Notification.                         |
| **Transport**          | STDIO, HTTP, SSE, Streamable HTTP.                                       |
| **Lifecycle**          | Init → Discovery → Execution → Termination.                              |
| **SDKs**               | Official TS + Python SDKs from Anthropic.                                |
| **Implementations**    | Hosts like ChatGPT, Claude; Servers via Gradio, Continue, custom Python. |

---
