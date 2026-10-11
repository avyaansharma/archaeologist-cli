# 🏺 Archaeologist CLI

<p align="center">
  <img src="docs/images/cli_ask.svg" alt="Archaeologist CLI — Forensic Causal Investigation in Your Terminal" width="100%" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+" /></a>
  <a href="https://rich.readthedocs.io/"><img src="https://img.shields.io/badge/CLI-Typer_%2B_Rich-10B981.svg?style=for-the-badge&logo=terminal&logoColor=white" alt="Typer + Rich CLI" /></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Google_Gemini-3.5_Flash-4285F4.svg?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini" /></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-Agentic_RAG-FF6F61.svg?style=for-the-badge" alt="LangGraph" /></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Model_Context_Protocol-000000.svg?style=for-the-badge" alt="MCP Server" /></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Qdrant-Hybrid_Vector_Store-DC2626.svg?style=for-the-badge" alt="Qdrant" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT" /></a>
</p>

---

### Autonomous Forensic Code Intelligence — Discover *Why* Code Changed, Directly in Your Terminal

> **Traditional Code RAG and GitHub Copilot explain *what* code does today.**  
> **Archaeologist CLI is an autonomous forensic AI agent that uncovers *why* code exists by mining the temporal causal graph — git commit diffs, AST symbol line mappings, PR discussions, linked issues, and bidirectional revert histories.**

---

## 🖥️ Terminal UI Showcase

Archaeologist CLI delivers a first-class terminal experience powered by [Rich](https://rich.readthedocs.io/) and [Typer](https://typer.tiangolo.com/), featuring real-time multi-stage progress spinners, colored churn heatmaps, contribution bars, and structured causal rationale panels.

### 1. Zero-Cost AST Structural Ingestion Engine
Ingests months or years of git commit history, AST symbols, and linked GitHub PR reviews at zero LLM cost during ingestion using AST structural diff footprints:

<p align="center">
  <img src="docs/images/cli_ingest.svg" alt="Archaeologist CLI — Ingestion Progress" width="95%" />
</p>

---

### 2. Forensic Causal Query (`archaeologist ask`)
Performs multi-hop LangGraph retrieval across commits, PRs, and incident reports with Just-In-Time (JIT) query-time diff expansion to explain the historical rationale behind any architectural decision:

<p align="center">
  <img src="docs/images/cli_ask.svg" alt="Archaeologist CLI — Forensic Causal Investigation" width="95%" />
</p>

---

### 3. Hotspots & Churn Visualizer (`archaeologist hotspots`)
Ranks high-risk files by modification frequency and visualizes churn velocity directly in your terminal:

<p align="center">
  <img src="docs/images/cli_hotspots.svg" alt="Archaeologist CLI — Hotspot Files & Churn" width="95%" />
</p>

---

### 4. Ownership & Bus Factor Risk Analyzer (`archaeologist ownership`)
Identifies maintainer contribution distributions and alerts you when a single engineer controls critical modules:

<p align="center">
  <img src="docs/images/cli_ownership.svg" alt="Archaeologist CLI — Code Ownership & Bus Factor" width="95%" />
</p>

---

### 5. Temporal Change Coupling (`archaeologist coupling`)
Discovers hidden dependencies and co-change patterns between files that frequently mutate together:

<p align="center">
  <img src="docs/images/cli_coupling.svg" alt="Archaeologist CLI — Temporal Coupling" width="95%" />
</p>

---

## 📌 Table of Contents

- [Key Use Cases](#-key-use-cases)
- [Why Archaeologist CLI is Different](#-why-archaeologist-cli-is-different)
- [Terminal CLI Workflows & Quickstart](#-terminal-cli-workflows--quickstart)
- [CLI Command Reference](#-cli-command-reference)
- [Native Model Context Protocol (MCP) Server](#-native-model-context-protocol-mcp-server)
- [Forensic Pipeline Architecture](#-forensic-pipeline-architecture)
- [Benchmark Results & Leaderboard](#-benchmark-results--leaderboard)
- [Production Hardening & Concurrency](#-production-hardening--concurrency)
- [License](#-license)

---

## 🎯 Key Use Cases

| Use Case | How Archaeologist CLI Solves It |
| :--- | :--- |
| **Onboarding to Legacy Repositories** | Answers questions like *"Why is this mutex lock here?"* or *"Why did we move away from `urllib3` defaults?"* by retrieving the original PR review debates and incident reports. |
| **Root Cause & Regression Forensics** | Connects broken lines directly to the offending commit, subsequent reverts (`reverts_sha`), and the specific author discussions that motivated the change. |
| **Safe Refactoring & Dead Code Pruning** | Warns developers before modifying code that was previously added to fix subtle production outages, edge-case race conditions, or security advisories. |
| **Team Ownership & Bus Factor Auditing** | Calculates per-file author distributions, identifies single-maintainer vulnerabilities, and maps temporal change coupling (files frequently edited together). |

---

## 🌟 Why Archaeologist CLI is Different

Standard developer tools inspect **only current code snapshots**:
- **Copilot / Cursor**: Explains the syntax of files currently open in your IDE.
- **Traditional Vector RAG**: Chunks static files, remaining completely blind to the historical context, rejected alternatives, and outage reports that shaped the code.

```text
src/services/user.py (fetchUser)
   │
   ├── Line Diff Mapping ──► Commit 8f31a2 ("Handle transient upstream 503 failures")
   │                            │
   │                            ├── Linked PR #421 ──► Code Review Comments (@tomchristie)
   │                            │                         │
   │                            └── Linked Issue #389 ──► Production Outage Incident Report
   │
   └── Bidirectional Revert ──► Commit 4a12c8 ("Initial single-pass request")
```

| Capability | Naive Vector RAG | Copilot / Cursor | Archaeologist CLI |
| :--- | :---: | :---: | :---: |
| **Analyzes Current Code (*What*)** | ✅ | ✅ | ✅ |
| **Mines Git History (*Why*)** | ❌ | ❌ | ✅ |
| **Zero-Cost AST Diff Footprints** | ❌ | ❌ | ✅ *(Zero API calls at ingest)* |
| **Local Vector Search (ONNX / FastEmbed)** | ❌ | ❌ | ✅ *(384-dim HNSW, CPU-local)* |
| **Just-In-Time Diff Expansion** | ❌ | ❌ | ✅ *(Full diffs on demand)* |
| **Multi-Channel RRF Ranking** | ❌ | ❌ | ✅ *(Exact, Dense, BM25, Fallback)* |
| **AST Symbol-Aware Line Mapping** | ❌ | Partial | ✅ *(Class & Method Level)* |
| **Bidirectional Revert Tracking** | ❌ | ❌ | ✅ *(`reverts_sha` ↔ `superseded_by`)* |
| **PR & Issue Discussion Traversal** | ❌ | ❌ | ✅ *(Cross-Link Graph)* |
| **Anti-Hallucination Fact Verification** | ❌ | ❌ | ✅ *(Deterministic Judge Node)* |
| **Native MCP stdio Protocol** | ❌ | ❌ | ✅ *(Claude Desktop / Cursor)* |

---

## 🚀 Terminal CLI Workflows & Quickstart

### Prerequisites
- **Python 3.11+**
- **Git** installed and available in your `PATH`
- **uv** (recommended) or `pip`

---

### 1. Installation

Install Archaeologist CLI using `uv` or `pip`:

```bash
# Clone the repository
git clone https://github.com/avyaansharma/archaeologist-cli.git
cd archaeologist-cli

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

---

### 2. Interactive Setup Wizard

Run the interactive setup wizard to configure your environment:

```bash
archaeologist setup
```

The wizard prompts for:
- **`GEMINI_API_KEY`**: Powers agent causal reasoning and verification judge ([Get a free key](https://aistudio.google.com/app/apikey)).
- **`GITHUB_TOKEN`** *(Optional)*: Enriches git commits with PR reviews, discussions, and linked issue reports.

You can also export them directly in your shell or `.env`:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GITHUB_TOKEN="ghp_your_token_here"  # Optional
```

---

### 3. Ingest Your Repository

Ingest any local repository into SQLite and local HNSW vector store:

```bash
# Standard mode: Ingest the last 6 months of commits (interactive window picker)
archaeologist ingest . --window 6m

# Full mode: Ingest complete commit history and link remote GitHub PRs/issues
archaeologist ingest . --window full --repo-url https://github.com/owner/repo

# Re-embed all chunks and recreate vector collection
archaeologist ingest . --reembed

# Optional: Enable legacy LLM diff summarization at ingest time (default: False, zero-cost AST footprints)
archaeologist ingest . --llm-summaries
```

---

### 4. Forensic Investigation Workflows

```bash
# 🔍 Ask why an architectural decision was made
archaeologist ask "Why was retry logic added to fetchUser?"

# 🧐 Explain why a specific line or range of code exists
archaeologist why src/auth.py:42

# 📊 Check repository index health and forensic statistics
archaeologist status

# 🔥 Inspect code churn hotspots and high-risk files (with optional --json)
archaeologist hotspots --top-n 10 --json

# 👥 Audit team code ownership and bus factor vulnerabilities
archaeologist ownership

# 🔗 Identify temporal coupling (files that frequently mutate together)
archaeologist coupling --min-co-commits 5

# 🧬 Trace the full commit lineage of a specific AST symbol
archaeologist symbol-history "UserService"
```

---

## 🛠 CLI Command Reference

| Command | Arguments / Flags | Description |
| :--- | :--- | :--- |
| **`setup`** *(alias: `init`)* | None | Interactive terminal wizard to configure API keys and persistent preferences. |
| **`status`** | `[--repo]` `[--json]` | Displays index health, total commits, vector collection count, and chunk stats. |
| **`ingest`** | `[REPO_PATH]` `--window` `--quick` `--since` `--repo-url` `--reembed` `--llm-summaries` | Ingests git commits, diff footprints, AST symbols, revert links, and GitHub PRs/issues into SQLite + Qdrant. |
| **`ask`** | `QUESTION` `[--repo]` `[--db-path]` | Executes the autonomous LangGraph multi-hop retrieval and self-verification agent loop. |
| **`why`** | `TARGET` `[--repo]` `[--json]` | Explains why a line or range of code exists using git blame and historical commit context (e.g. `src/auth.py:42`). |
| **`hotspots`** | `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Ranks files by modification frequency with formatted churn visualizer bars or JSON. |
| **`ownership`** | `[--file-path]` `[--repo]` `[--db-path]` `[--json]` | Computes per-author contribution percentages and flags single-maintainer Bus Factor risks (>60% dominance). |
| **`coupling`** | `[--min-co-commits]` `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Detects pairs of files that change together across commits with temporal strength metrics. |
| **`symbols`** | `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Lists extracted AST code symbols (functions, classes, methods) ranked by mutation count. |
| **`symbol-history`** | `SYMBOL_NAME` `[--repo]` `[--db-path]` `[--json]` | Traces every commit and diff hunk that modified a specific AST code symbol over time. |
| **`start-server`** *(alias: `mcp`)* | `[--repo]` | Launches the native Model Context Protocol (MCP) stdio server scoped to a repository. |
| **`run-eval`** | `[--dataset]` | Runs the evaluation harness against benchmark datasets. |

> **Note on Language Support**: AST symbol parsing (`symbols`, `symbol-history`) parses Python (`.py`) abstract syntax trees. For all other programming languages (TypeScript, JavaScript, Go, Rust, Java, C++, etc.), full git commit histories, unified diff hunks, author ownership distributions, temporal change coupling, and GitHub PR/issue cross-references are 100% indexed and fully searchable.

---

## 🔌 Native Model Context Protocol (MCP) Server

Archaeologist CLI includes a native MCP stdio server that connects your local codebase forensic graph directly into **Claude Desktop**, **Cursor**, or any MCP-compatible environment.

### Claude Desktop Configuration (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "archaeologist": {
      "command": "uv",
      "args": [
        "run",
        "archaeologist",
        "mcp",
        "--repo",
        "C:/path/to/target/repo"
      ],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key",
        "GITHUB_TOKEN": "your_github_token"
      }
    }
  }
}
```

### Cursor MCP Configuration (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "archaeologist": {
      "command": "archaeologist",
      "args": ["mcp", "--repo", "."],
      "env": {
        "GEMINI_API_KEY": "your_gemini_api_key"
      }
    }
  }
}
```

### Registered Forensic MCP Tools:
- `ask(question, repo_id)`: Runs the autonomous LangGraph multi-hop retrieval and self-verification agent.
- `search_history(query, file_path, date_from, date_to, repo_id)`: Hybrid RRF search over commits, PRs, and issues.
- `blame_explain(file_path, line_start, line_end)`: Explains the causal rationale behind specific lines of code with JIT diff expansion.
- `find_related_discussion(ref, repo_id)`: Cross-references commit SHAs, PRs, and issues.
- `repo_hotspots(top_n, repo_id)`: Ranks files by commit frequency and churn.
- `repo_ownership(file_path, repo_id)`: Analyzes author contribution distribution and flags bus factor risks.
- `change_coupling(min_co_commits, top_n, repo_id)`: Identifies file pairs that frequently change together.
- `symbol_history(symbol_name, repo_id)`: Traces every commit that modified a specific AST class or function.

---

## 🤖 Forensic Pipeline Architecture

Archaeologist CLI employs a two-tier forensic architecture that eliminates LLM quota exhaustion at ingest time while maximizing synthesis depth at query time:

```text
======================= INGESTION TIER (Zero-Cost Local) =======================
  Git Commit History ──► AST Structural Diff Footprint Extraction
          │                       │
          ▼                       ▼
  Local SQLite (FTS5/BM25)  FastEmbed (ONNX 384-dim) ──► Qdrant HNSW Index
          │                       │
          └───────────────────────┴───────────────────────────────┐
                                                                  ▼
======================= QUERY TIER (Agentic Multi-Hop) =========================
  User Question ──► Candidate Forensics Discovery (Exact SHA/PR/#Issue)
                          │
                          ▼
                    Decompose & Plan
                          │
                          ▼
            Multi-Channel Reciprocal Rank Fusion (RRF)
     ┌────────────────────┬────────────────────┬────────────────────┐
     │ Exact Channel      │ Dense Vector (HNSW)│ Sparse BM25        │ Keyword Fallback
     │ (Weight: 2.5)      │ (Weight: 1.0)      │ (Weight: 1.0)      │ (Weight: 0.2)
     └────────────────────┴────────────────────┴────────────────────┘
                          │
                          ▼
                Cross-Link Graph Traversal (Commit ↔ PR ↔ Issue)
                          │
                          ▼
                JIT Query-Time Diff Expansion (Fetches unified diffs on-demand)
                          │
                          ▼
                Self-Verification Judge (4,000 Token Budget + Quota Circuit-Breaker)
                          │
                          ▼
                Grounded Forensic Synthesis with Exact Provenance Citations
```

1. **Zero-Cost AST Structural Patch Ingestion**: Commits are indexed using author metadata, commit messages, touched file paths, and exact AST symbol signatures without making LLM diff summarization API calls.
2. **Multi-Channel Reciprocal Rank Fusion (RRF)**: Disentangles exact token matches from semantic vectors. Exact commit SHAs and PR numbers receive a prioritized boost without allowing fallback keyword matches to displace high-similarity vector hits.
3. **Just-In-Time (JIT) Diff Expansion**: Unified diffs are extracted directly from git blobs during synthesis and `blame_explain`, providing complete contextual patch lines exactly when needed.
4. **Self-Verifying Reflection Loop with Quota Circuit-Breaker**: An independent fact-checker node validates claims against evidence under a 4,000-token evidence budget. If quota limits (429) occur, the circuit breaker bypasses retries to ensure graceful degradation.

---

## 🏆 Benchmark Results & Leaderboard

Evaluated using a deterministic mathematical scoring harness combining LLM-as-a-Judge semantic entailment with exact commit SHA / PR citation verification:

$$\text{Grounded Accuracy} = 0.70 \times \text{Atomic Fact Entailment} + 0.30 \times \text{True Citation } F_1$$

<p align="center">
  <img src="docs/images/leaderboard.png" alt="Evaluation Leaderboard Benchmark" width="90%" />
</p>

| Target Repository | Scale | Grounded Accuracy | Atomic Entailment | True Citation $F_1$ | Key Highlights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`psf/requests`** | **7,163 Chunks** (6,490 Commits) | **93.27%** | **100.00%** | **77.55%** | **100% Fact Entailment** across all questions (`Session.send` = 97.27%, `raise_for_status` = 100%) |
| **`pallets/flask`** | **1,390 Chunks** (673 Commits) | **88.99%** | **88.89%** | **89.22%** | `ContextVar` isolation = 100%, Click CLI = 100%, `full_dispatch_request` = 96.67% |
| **`BoboTiG/python-mss`** | **2,514 Chunks** (1,053 Commits) | **86.77%** | **86.67%** | **87.00%** | Revert `2d24115` = 100%, Xlib per-object lock = 100%, `memoryview` = 100% |

---

## 🛡️ Production Hardening & Concurrency

Archaeologist CLI is engineered for high-concurrency production environments and developer workstations:

- **ContextVar Request Isolation**: Web and API servers use `contextvars.ContextVar` for coroutine-scoped isolation, preventing environment variable race conditions across concurrent async requests.
- **SQLite WAL Mode & Connection Pooling**: Automatic `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=10000;` on all database connections prevent lock contention and disposal thrashing.
- **Fail-Loud Vector Writer Locks**: When the local Qdrant directory is accessed by an active MCP server or UI session, writes fail cleanly with `VectorStoreLockedError` rather than silently losing embeddings in volatile memory.
- **Rate-Limit Resilient Agent**: Automatic quota circuit-breaking prevents unbounded retry loops when Google Gemini 429 rate limits are encountered.
- **Tracked Git File Isolation**: Code ingestion lists files with `git ls-files`, strictly ignoring untracked `.env` files, build caches, and test artifacts.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
