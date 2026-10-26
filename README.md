# 🏺 Archaeologist CLI

<p align="center">
  <video src="docs/images/archaeologist_demo.mp4" width="100%" controls autoplay loop muted></video>
</p>


<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11_%7C_3.12-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11 | 3.12" /></a>
  <a href="https://github.com/avyaansharma/archaeologist-cli/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/CI-Passing_(236_tests)-brightgreen.svg?style=for-the-badge&logo=github-actions&logoColor=white" alt="CI: 236 Tests Passing" /></a>
  <a href="https://www.sqlite.org/"><img src="https://img.shields.io/badge/SQL-SQLite_%2B_Recursive_CTEs-003B57.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite + Recursive CTEs" /></a>
  <a href="https://rich.readthedocs.io/"><img src="https://img.shields.io/badge/CLI-Typer_%2B_Rich-10B981.svg?style=for-the-badge&logo=terminal&logoColor=white" alt="Typer + Rich CLI" /></a>
  <a href="https://github.com/Qiskit/rustworkx"><img src="https://img.shields.io/badge/Topology-rustworkx-DEA584.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rustworkx" /></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/Agent-LangGraph-FF6F61.svg?style=for-the-badge" alt="LangGraph" /></a>
  <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP-Model_Context_Protocol-000000.svg?style=for-the-badge" alt="MCP Server" /></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Qdrant-Hybrid_Vector_Store-DC2626.svg?style=for-the-badge" alt="Qdrant" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT" /></a>
</p>

---

### Autonomous Forensic Code Intelligence — Discover *Why* Code Changed, Directly in Your Terminal

> **Traditional Code RAG and GitHub Copilot explain *what* code does today.**  
> **Archaeologist CLI is an autonomous forensic AI agent that uncovers *why* code exists by mining the temporal causal graph — git commit diffs, AST symbol line mappings, PR discussions, linked issues, and bidirectional revert histories.**

> 💡 **Independent CLI Tool**: Archaeologist CLI (`archaeologist` / `arch`) is a dedicated, standalone command-line forensic intelligence tool and native Model Context Protocol (MCP) server. It operates completely on local git repositories with zero web servers, browser dashboards, or external platform dependencies.

---

## 🖥️ Terminal UI Showcase

Archaeologist CLI delivers a first-class terminal experience powered by [Rich](https://rich.readthedocs.io/) and [Typer](https://typer.tiangolo.com/), featuring real-time multi-stage progress spinners, colored churn heatmaps, contribution bars, and structured causal rationale panels.

### 1. Zero-Cost AST Structural Ingestion Engine
Ingests years of git commit history, AST symbols, and linked GitHub PR reviews at zero LLM cost using AST structural diff footprints:

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
- [Two-Tier Hybrid Causal Graph Architecture](#-two-tier-hybrid-causal-graph-architecture)
- [Hybrid Retrieval & 4-Stage Deterministic Verification](#-hybrid-retrieval--4-stage-deterministic-verification)
- [Terminal CLI Workflows & Quickstart](#-terminal-cli-workflows--quickstart)
- [CLI Command Reference](#-cli-command-reference)
- [Native Model Context Protocol (MCP) Server](#-native-model-context-protocol-mcp-server)
- [Benchmark Results & Performance Metrics](#-benchmark-results--performance-metrics)
- [Production Hardening & Reliability](#-production-hardening--reliability)
- [License](#-license)

---

## 🎯 Key Use Cases

| Use Case | How Archaeologist CLI Solves It |
| :--- | :--- |
| **Onboarding to Legacy Repositories** | Answers questions like *"Why is this mutex lock here?"* or *"Why did we move away from `urllib3` defaults?"* by retrieving original PR review debates, architectural discussions, and incident reports. |
| **Root Cause & Regression Forensics** | Connects broken lines directly to the offending commit, subsequent reverts (`reverts_sha`), and the specific author discussions that motivated the change. |
| **Safe Refactoring & Dead Code Pruning** | Warns developers before modifying code that was previously added to fix subtle production outages, edge-case race conditions, or security advisories. |
| **Architectural Hotspot & Centrality Analysis** | Employs Rust-accelerated whole-graph algorithms (PageRank, Betweenness Centrality) to identify structural bottlenecks and coupled subsystems. |
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
| **Two-Tier Hybrid Causal Graph** | ❌ | ❌ | ✅ *(SQLite CTEs + rustworkx)* |
| **Zero-Cost AST Diff Footprints** | ❌ | ❌ | ✅ *(Zero API calls at ingest)* |
| **Local Vector Search (ONNX / FastEmbed)** | ❌ | ❌ | ✅ *(768-dim Code ONNX, CPU-local)* |
| **Just-In-Time Diff Expansion** | ❌ | ❌ | ✅ *(Full diffs on demand)* |
| **4-Channel Isolated RRF Fusion** | ❌ | ❌ | ✅ *(Exact, Dense, BM25, Symbol)* |
| **AST Symbol-Aware Line Mapping** | ❌ | Partial | ✅ *(Class & Method Level)* |
| **Bidirectional Revert Tracking** | ❌ | ❌ | ✅ *(`reverts_sha` ↔ `superseded_by`)* |
| **PR & Issue Discussion Traversal** | ❌ | ❌ | ✅ *(Cross-Link Graph)* |
| **4-Stage Deterministic Verification Engine** | ❌ | ❌ | ✅ *(SQLite Tuple & AST Grounding, 25x Faster)* |
| **Native MCP stdio Protocol** | ❌ | ❌ | ✅ *(Claude Desktop / Cursor)* |

---

## 🏛️ Two-Tier Hybrid Causal Graph Architecture

Archaeologist CLI decouples instant multi-hop edge traversal from complex global topology analysis using a **Two-Tier Hybrid Architecture**:

```text
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 1: RELATIONAL & MULTI-HOP RETRIEVAL TIER                     │
│               SQLite `causal_edges` + Native Recursive CTEs (C-Engine)                │
│                                                                                       │
│   • Sub-millisecond traversal (< 2ms) across multi-hop dependency chains              │
│   • Eliminates runtime string matching and N+1 database queries in follow_links       │
│   • Edges: `resolves`, `merges`, `reverts`, `superseded_by`, `modifies`               │
│   • Strict multi-tenancy scoping via composite `(repo_id, source_id, target_id)`     │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
                                           │ Synchronized on Ingestion
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 2: ANALYTICAL & WHOLE-GRAPH TOPOLOGY TIER                    │
│                      Rust-Accelerated `rustworkx.PyDiGraph` (Rust)                    │
│                                                                                       │
│   • Whole-repository PageRank on critical AST symbols and merge commits               │
│   • Betweenness Centrality on architecturally coupled bottleneck files                │
│   • Sub-100ms global graph computation with ~80% less memory than NetworkX            │
│   • Thread-safe Rust execution releasing Python GIL                                   │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Storage & Multi-Hop Retrieval Tier (SQLite CTEs)
Edges discovered during git parsing, AST symbol analysis, PR discussions, and issue resolutions are materialized into an indexed `causal_edges` relational table:
- **Zero-Hop Lookups**: Directly resolves linked PRs, merged commits, and modified functions.
- **Recursive Transitive Traversals**: Resolves complex chains (e.g. `Issue #101` ➔ `PR #501` ➔ `Commit a1b2c3` ➔ `Symbol auth::validate_token` ➔ `Revert d4e5f6`) in under 2ms using a single SQLite `WITH RECURSIVE` CTE query.

### 2. Analytical & Topology Tier (`rustworkx`)
For repository-wide architecture metrics, edges are loaded into `rustworkx.PyDiGraph`:
- **Symbol PageRank**: Computes importance scores across all classes and functions based on temporal dependency propagation.
- **Betweenness Centrality**: Pinpoints bottleneck files and bridge components that control the flow of system mutations.
- **Zero Cross-Platform Friction**: Ships with pre-compiled wheels for Linux, macOS, and Windows with zero C++ toolchain requirements.

Run whole-graph topology analysis from the terminal:
```bash
archaeologist graph-topology --top-n 10
```

---

## 🔎 Hybrid Retrieval & 4-Stage Deterministic Verification

Archaeologist CLI combines a four-channel retrieval pipeline with a **4-Stage Hybrid Deterministic Verification Engine** (`verification_engine.py`) that validates claims mathematically against SQLite and AST footprints in **~50ms** while eliminating redundant LLM Judge calls:

```text
                                  User Query
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
        Dense Vector (HNSW)      Sparse BM25          Exact Candidates
       (FastEmbed / Gemini)   (Rank-BM25 + Dates)    (SHAs, PRs, Issues)
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      ▼
                        Isolated Symbol AST Routing
                                      │
                                      ▼
                     4-Channel Reciprocal Rank Fusion
       RRF(d) = Σ w_c / (k + rank_c(d))  with High-Confidence Top-1 Preservation
                                      │
                                      ▼
                     Recursive CTE Transitive Link Expansion
          (`follow_links` discovers PRs/SHAs with 0.92^depth score inheritance)
                                      │
                                      ▼
           STAGE 1: Pre-Draft Causal Link Auto-Hydration & Stratification
        (Hydrates missing linked PRs/Reverts from SQLite; top-3 per sub-question)
                                      │
                                      ▼
             STAGE 2: SQLite & Chunk Relational Tuple Verification
     (Checks SHA/PR existence, Commit ↔ File/Symbol ownership, Reverts & Chronology)
                                      │
                                      ▼
          STAGE 3: Structural Code Identifier Grounding & Confidence Gate
      (Short-Circuits errors or Fast-Passes verified drafts in ~50ms, 0 Judge tokens)
                                      │
                                      ▼
           STAGE 4: Grounded JIT Diff Synthesis & Post-Synthesis Guardrail
         (24,000-char JIT diff budget + deterministic PR/Issue citation guardrail)
```

### 1. Multi-Channel Isolation in RRF
- **Exact Channel** (`weight=2.5`): Direct matches on commit SHAs, PR numbers (`#421`), and issue identifiers.
- **Dense Vector Channel** (`weight=1.0`): Semantic retrieval via 768-dimensional code-specialized FastEmbed ONNX (`jinaai/jina-embeddings-v2-base-code`), 384-dimensional BGE, or 3072-dimensional Gemini embeddings. High-confidence top-1 vector hits are safeguarded against score inversion.
- **Sparse Lexical Channel** (`weight=1.0`): Rank-BM25 with token-shape normalization, exact code identifier boosting, and UTC date bounds.
- **Symbol Channel Isolation** (`weight=0.2`): Synthetic symbol scores are kept in an isolated channel, preventing uncalibrated pseudo-scores from displacing authentic semantic hits.

### 2. Causal Link Score Inheritance & Stratified Sub-Question Budgeting
- **Parent-Inherited Transitive Scoring**: When `follow_links` expands multi-hop causal edges (`resolves`, `merges`, `reverts`, `superseded_by`), discovered nodes inherit decayed scores from their seed commit (`max(0.012, parent_score * 0.92^depth)`), ensuring critical PR discussions and incident reports rank immediately alongside the commits they explain.
- **Stratified Multi-Sub-Question Selection**: For complex queries decomposed into multiple sub-questions, `verify` and `synthesize` interleave the top-3 highest-scoring chunks from each sub-question before global RRF backfilling, preventing any single sub-question from crowding out others within the prompt window.

### 3. 4-Stage Deterministic Verification Engine (`verification_engine.py`)
Instead of relying on an expensive, confirmation-biased secondary LLM call to fact-check drafts, Archaeologist CLI verifies claims deterministically against the repository's relational ground truth:
- **Stage 1 — Pre-Draft Causal Link Auto-Hydration**: Scans top retrieved chunks and automatically hydrates missing linked PRs, Issues, or reverted commits (`reverts_sha`) directly from SQLite (scoped by `repo_id`) in `<12ms` before drafting.
- **Stage 2 — Relational Tuple & Provenance Verifier**: Builds a unified `EvidenceIndex` from retrieved chunks and SQLite (`Commit`, `PullRequest`, `Issue`) to verify clause-scoped **Commit ↔ File** and **Commit ↔ AST Symbol** ownership (catching cross-commit Chimera hallucinations), **directional revert integrity** (`reverts_sha` vs. `superseded_by_sha`), and **chronological commit sequence** (`authored_date`).
- **Stage 3 — Code Identifier Grounding & Confidence-Gated LLM Bypass**: Verifies backticked and `snake_case` code identifiers against the evidence index while filtering out English hex words (`feedback`, `defaced`, `facade`) and numeric dates. Immediately **short-circuits** (`0` Judge LLM tokens) on any Stage 2/3 contradiction and **fast-passes** (`0` Judge LLM tokens) when citations and code identifiers are deterministically verified—reducing verification latency by **96.02% (25.1x faster)**.
- **Stage 4 — Post-Synthesis Provenance & Citation Guardrail**: Inspects the final synthesized answer (after 24,000-character Just-In-Time `git show <sha>` diff expansion) to flag any post-draft hallucinated references and deterministically append any omitted verified PR/Issue links from top evidence.

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
# Clone the repository (HTTPS or SSH)
git clone https://github.com/avyaansharma/archaeologist-cli.git
# or: git clone git@github.com:avyaansharma/archaeologist-cli.git

cd archaeologist-cli

# Install dependencies and sync locked environment
uv sync --locked

# Or install with pip
pip install -e .
```

Both `archaeologist` and its shorthand alias `arch` are immediately accessible in your terminal:
```bash
archaeologist --help
# Or shorthand
arch --help
```

---

### 2. Interactive Setup Wizard & Environment Options

Run the interactive setup wizard to configure your environment:

```bash
archaeologist setup
```

The wizard prompts for:
- **`GEMINI_API_KEY`**: Powers agent causal reasoning and verification judge ([Get a free key](https://aistudio.google.com/app/apikey)).
- **`GITHUB_TOKEN`** *(Optional)*: Enriches git commits with PR reviews, discussions, and linked issue reports.

#### Option A: Cloud Google Gemini (Default)
Export keys directly in your shell or `.env`:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GITHUB_TOKEN="ghp_your_token_here"  # Optional
```

#### Option B: 100% Offline / Local Models (Ollama, vLLM, LM Studio)
Archaeologist CLI can operate completely offline with zero external API calls:
```bash
# Point to your local OpenAI-compatible inference endpoint (e.g. Ollama)
export LOCAL_LLM_URL="http://localhost:11434/v1"
export LOCAL_LLM_MODEL="qwen2.5-coder:7b"  # or llama3.2, deepseek-coder
export USE_FASTEMBED=true                  # Local 768-dim code ONNX CPU embeddings ($0.00 cost)
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

# 📊 Check repository index health and multi-tenant forensic statistics
archaeologist status

# 🕸️ Compute whole-repository graph topology metrics (PageRank & Betweenness Centrality)
archaeologist graph-topology --top-n 10

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
| **`status`** | `[--repo]` `[--db-path]` `[--json]` | Displays index health, commit counts, vector point counts, and tenant metadata. |
| **`ingest`** | `[REPO_PATH]` `--window` `--quick` `--since` `--repo-url` `--reembed` `--llm-summaries` | Ingests git commits, diff footprints, AST symbols, revert links, and GitHub PRs/issues into SQLite + Qdrant. |
| **`ask`** | `QUESTION` `[--repo]` `[--db-path]` | Executes autonomous LangGraph multi-hop retrieval with CTE graph expansion and self-verification. |
| **`why`** | `TARGET` `[--repo]` `[--json]` | Explains why a line or range of code exists using git blame and historical commit context (e.g. `src/auth.py:42`). |
| **`graph-topology`** | `[--repo]` `[--top-n]` `[--json]` | Analyzes repository topology with `rustworkx`, computing PageRank and betweenness centrality. |
| **`hotspots`** | `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Ranks files by modification frequency with formatted churn visualizer bars or JSON. |
| **`ownership`** | `[--file-path]` `[--repo]` `[--db-path]` `[--json]` | Computes per-author contribution percentages and flags single-maintainer Bus Factor risks (>60% dominance). |
| **`coupling`** | `[--min-co-commits]` `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Detects pairs of files that change together across commits with temporal strength metrics. |
| **`symbols`** | `[--top-n]` `[--repo]` `[--db-path]` `[--json]` | Lists extracted AST code symbols (functions, classes, methods) ranked by mutation count. |
| **`symbol-history`** | `SYMBOL_NAME` `[--repo]` `[--db-path]` `[--json]` | Traces every commit and diff hunk that modified a specific AST code symbol over time. |
| **`start-server`** *(alias: `mcp`)* | `[--repo]` | Launches the native Model Context Protocol (MCP) stdio server scoped to a repository. |
| **`run-eval`** | `[--dataset]` | Runs the evaluation harness against benchmark datasets. |

---

## 🔌 Native Model Context Protocol (MCP) Server

Archaeologist CLI includes a native MCP stdio server that connects your local git repository's forensic causal graph directly into **Claude Desktop**, **Cursor**, or any MCP-compatible environment.

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
- `ask(question, repo_id)`: Autonomous LangGraph multi-hop retrieval, CTE expansion, and verification.
- `graph_topology(top_n, repo_id)`: Whole-graph PageRank and betweenness centrality via `rustworkx`.
- `search_history(query, file_path, date_from, date_to, repo_id)`: Hybrid RRF search over commits, PRs, and issues.
- `blame_explain(file_path, line_start, line_end)`: Causal rationale behind code lines with JIT diff expansion.
- `find_related_discussion(ref, repo_id)`: Cross-references commit SHAs, PRs, and issues.
- `repo_hotspots(top_n, repo_id)`: Ranks files by commit frequency and churn.
- `repo_ownership(file_path, repo_id)`: Author contribution distribution and bus factor analysis.
- `change_coupling(min_co_commits, top_n, repo_id)`: File co-mutation patterns.
- `symbol_history(symbol_name, repo_id)`: Commit lineage for an AST class or function.

---

## 🏆 Benchmark Results & Performance Metrics

Evaluated using a **reproducible hybrid evaluation harness** combining **deterministic provenance verification** (exact commit SHA existence checks against the git object tree and PR resolution in SQLite) with **calibrated LLM-as-a-Judge semantic entailment scoring** ($T=0.0$ structured proposition verification to minimize evaluation variance):

$$\text{Grounded Accuracy} = 0.70 \times \text{Atomic Fact Entailment} + 0.30 \times \text{True Citation } F_1$$

> ℹ️ **Evaluation Protocol**: Citation accuracy is mathematically verified against the local repository's git object graph and SQLite entity tables (100% deterministic). Atomic fact entailment evaluates whether each ground-truth proposition is entailed by the synthesized rationale using low-temperature structured verification.

### 📊 Comparative Baseline Results

| Retrieval Paradigm | Grounded Accuracy | Atomic Entailment | Citation $F_1$ | Multi-Hop Causal Chains | AST Line Attribution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Lexical Search** | 38.5% | 46.2% | 20.5% | ❌ Disconnected | ❌ None |
| **Naive Vector RAG** *(Chunk-level)* | 44.2% | 51.0% | 28.3% | ❌ Disconnected | ❌ Token-chunk only |
| **Git Blame + Log Keyword** | 51.3% | 58.7% | 34.0% | ❌ Surface-level | ⚠️ Line-level only |
| **Archaeologist CLI (Ours)** | **89.7%** *(Avg)* | **91.9%** | **84.6%** | ✅ **Recursive CTEs** | ✅ **Class & Method AST** |

### 🔬 Repository Benchmark Breakdown

<p align="center">
  <img src="docs/images/leaderboard.png" alt="Evaluation Leaderboard Benchmark" width="90%" />
</p>

| Target Repository | Scale | Grounded Accuracy | Atomic Entailment | True Citation $F_1$ | Key Highlights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`psf/requests`** | **7,163 Chunks** (6,490 Commits) | **93.27%** | **100.00%** | **77.55%** | **100% Fact Entailment** across all questions (`Session.send` = 97.27%, `raise_for_status` = 100%) |
| **`pallets/flask`** | **1,390 Chunks** (673 Commits) | **88.99%** | **88.89%** | **89.22%** | `ContextVar` isolation = 100%, Click CLI = 100%, `full_dispatch_request` = 96.67% |
| **`BoboTiG/python-mss`** | **2,514 Chunks** (1,053 Commits) | **86.77%** | **86.67%** | **87.00%** | Revert `2d24115` = 100%, Xlib per-object lock = 100%, `memoryview` = 100% |

### ⚡ 4-Stage Deterministic Verification & Pipeline Performance (`psf/requests` Benchmark)

Head-to-head benchmark comparing the legacy LLM-Judge verification pipeline against the **4-Stage Deterministic Verification Engine** (`verification_engine.py`) and $O(\log C)$ link resolution on **`psf/requests`** (**6,490 commits, 1,465 PRs**):

| Metric | Legacy LLM-Judge Pipeline | 4-Stage Deterministic Engine | Verified Improvement |
| :--- | :---: | :---: | :---: |
| **Claim Verification Accuracy** *(Valid + Adversarial Suite)* | 80.00% *(8/10)* | **100.00%** *(10/10)* | **+20.00 pp** *(+25.0% relative)* |
| **Adversarial Hallucination Recall** *(Cross-commit, Revert & Chrono Inversions)* | 100.00% *(5/5)* | **100.00%** *(5/5)* | **Zero false negatives** + **Zero false positives** |
| **Verification Latency (Average per Query)** | 1,270.15 ms | **50.53 ms** | **-96.02% time** *(**25.1x faster**)* |
| **Verification Latency (10-Query Suite Total)** | 12,701.50 ms | **505.28 ms** | **-12,196.22 ms saved** |
| **Judge LLM API Calls & Token Cost** *(Cited / Flawed Drafts)* | 10 calls *(3,587 tokens)* | **0 calls** *(0 tokens)* | **-100.00% Judge LLM cost** *(**-20% to -50%** total agent calls)* |
| **Step 5 PR-to-Commit Link Resolution** *(1,465 PRs × 6,490 Commits)* | 8,891.50 ms *($O(P \cdot L \cdot C)$)* | **11.85 ms** *($O(1) + O(\log C)$ `bisect`)* | **-99.87% time** *(**750.5x faster**, 1,476/1,476 matches)* |

---

## 🔒 Privacy, Security & Zero-Cost Architecture

Archaeologist CLI is purpose-built with privacy-first and cost-conscious engineering invariants:

- **100% Local Git & Relational Storage**: All git history, AST symbol indices, and dependency edges are stored strictly inside your local workspace in SQLite (`.db`) and embedded Qdrant (`qdrant_storage/`). Zero repository source code or commit logs are uploaded to any external server during indexing or graph analysis.
- **Zero Ingestion Cost ($0.00)**: Ingestion uses AST structural patch footprint extraction and local CPU-accelerated FastEmbed embeddings (`jinaai/jina-embeddings-v2-base-code` 768-dim code & diff representations, with backwards compatibility for `BAAI/bge-small-en-v1.5` via ONNX Runtime). Indexing codebases requires zero LLM API calls, zero embeddings API tokens, and zero paid subscriptions.
- **Just-In-Time (JIT) Diff Expansion**: Full unified diffs are extracted on demand directly from git blobs only for candidate commits during query time, ensuring massive token savings and minimal external transmission.
- **100% Air-Gapped Local Inference Option**: Fully supports local, offline models (e.g. **Ollama**, **vLLM**, or **LM Studio**) via `LOCAL_LLM_URL="http://localhost:11434/v1"` and `USE_FASTEMBED=true`. Enables completely air-gapped forensic investigations in proprietary, classified, or zero-trust corporate codebases.

---

## 🛡️ Production Hardening & Reliability

Archaeologist CLI is engineered for high-concurrency developer environments and large-scale codebases:

- **Strict Multi-Tenancy Scoping**: `Commit`, `PullRequest`, `Issue`, `SymbolIndex`, and `CausalEdge` use composite primary keys `(repo_id, ...)`. Ingesting forks, worktrees, or shared commit histories never causes collisions or cross-tenant data leaks.
- **Zero-Config Vector Dimension Auto-Adaptation**: Read-only queries across CLI, Python API, and MCP tools automatically inspect `<repo>/.archaeologist/meta.json` and SQLite `RepoMeta` (or on-disk Qdrant collection dimensions: 384-dim BGE, 768-dim Jina Code ONNX, or 3072-dim Gemini) and parameterize the active `Embedder` without mutating process-wide `os.environ` state.
- **AST-Enriched File Snapshots & $O(\log C)$ PR Link Resolution**: Step 1b current-file snapshots automatically embed extracted AST class/function symbol tables alongside source content, while Step 5 PR-to-commit linkage uses $O(1)$ SHA/message lookup tables and $O(\log C)$ binary search (`bisect`) over author timestamps to scale cleanly to 50k+ commit repositories.
- **Fail-Fast Vector Lock Detection**: `VectorStore(writable=True)` checks local Qdrant directory locks at Step 0, preventing wasted ingestion time when another process holds write locks.
- **Incremental Re-Ingestion & Stale Point Purging**: Chunks use deterministic UUID5 content hashing. Unchanged files skip re-embedding, and retired file vectors are purged from Qdrant.
- **SQLite Strict Adoption, WAL Mode & Connection Pooling**: Enforces exclusive SQLite storage (ignoring stray ambient `postgresql://` shell variables) with automatic `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=10000;` to prevent lock contention during rapid multi-hop forensic queries.
- **Circuit-Breaker Protected Agent**: Automated quota circuit-breakers in `verify_node` prevent runaway retries during rate limit (429) events, falling back gracefully to evidence summaries.
- **Comprehensive Test Suite & CI Matrix**: 236 tests passing across Ubuntu and Windows under Python 3.11 and 3.12 with zero linter warnings.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
