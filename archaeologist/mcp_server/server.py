import os
import json
from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        from mcp.server import MCPServer as FastMCP

from archaeologist.mcp_server.tools import (
    search_history_tool,
    find_related_discussion_tool,
    blame_explain_tool,
    repo_hotspots_tool,
    repo_ownership_tool,
    change_coupling_tool,
    repo_symbols_tool,
    symbol_history_tool,
    ask_tool
)

# Initialize FastMCP Server
mcp = FastMCP("Codebase Archaeologist")

@mcp.tool()
def search_history(
    query: str,
    file_path: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    source_types: Optional[List[str]] = None
) -> str:
    """Hybrid search over commit/PR/issue history for the repository.
    
    Arguments:
      query: The keyword or semantic search term.
      file_path: Optional file path constraint (e.g. 'src/auth.py').
      date_from: Optional starting date filter (YYYY-MM-DD).
      date_to: Optional ending date filter (YYYY-MM-DD).
      source_types: List of sources to search: 'commit', 'pr', 'issue'.
    """
    results = search_history_tool(
        query=query,
        file_path=file_path,
        date_from=date_from,
        date_to=date_to,
        source_types=source_types
    )
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def blame_explain(
    file_path: str,
    line_start: int,
    line_end: int,
    repo_path: Optional[str] = None
) -> str:
    """Traces line history using git blame and returns causal explanations of those lines.
    
    Arguments:
      file_path: Path to the target file.
      line_start: Starting line number (1-indexed).
      line_end: Ending line number (1-indexed).
      repo_path: Optional local repository path (defaults to current directory).
    """
    configured_root = os.path.abspath(os.getenv("ARCHAEOLOGIST_REPO") or os.getenv("GIT_DETECTIVE_REPO", "."))
    if repo_path:
        candidate = os.path.abspath(repo_path)
        if candidate != configured_root and not candidate.startswith(configured_root + os.sep):
            path = configured_root
        else:
            path = candidate
    else:
        path = configured_root
    results = blame_explain_tool(
        repo_path=path,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end
    )
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def find_related_discussion(ref: str) -> str:
    """Given a commit SHA or PR/issue number, returns linked commits, PRs, and issue references.
    
    Arguments:
      ref: The reference to look up (e.g. commit SHA 'a1b2c3d', issue '#123', or PR '#456').
    """
    results = find_related_discussion_tool(ref)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def repo_hotspots(top_n: int = 15) -> str:
    """Ranks files by commit frequency (most frequently modified files / hotspots).
    
    Arguments:
      top_n: Number of top hotspot files to return.
    """
    results = repo_hotspots_tool(top_n=top_n)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def repo_ownership(file_path: Optional[str] = None) -> str:
    """Analyzes author contribution distribution per file to determine ownership and bus factor risk.
    
    Arguments:
      file_path: Optional path to inspect ownership for a specific file.
    """
    results = repo_ownership_tool(file_path=file_path)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def change_coupling(min_co_commits: int = 2, top_n: int = 15) -> str:
    """Finds pairs of files that frequently change together in the same commit (temporal coupling).
    
    Arguments:
      min_co_commits: Minimum number of co-commits required.
      top_n: Number of top coupled file pairs to return.
    """
    results = change_coupling_tool(min_co_commits=min_co_commits, top_n=top_n)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def repo_symbols(top_n: int = 20) -> str:
    """Lists extracted AST Code Symbols (classes, functions, methods) ranked by modification frequency.
    
    Arguments:
      top_n: Number of top AST symbols to list.
    """
    results = repo_symbols_tool(top_n=top_n)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def symbol_history(symbol_query: str) -> str:
    """Retrieves all commits that modified a specific AST Code Symbol (e.g. 'AuthService' or 'login').
    
    Arguments:
      symbol_query: Class or function name to search in AST symbol history.
    """
    results = symbol_history_tool(symbol_query=symbol_query)
    return json.dumps(results, indent=2, default=str)

@mcp.tool()
def ask(question: str) -> str:
    """Answer a causal 'why' question about the codebase using full agentic multi-hop retrieval.
    
    Arguments:
      question: The causal question to ask (e.g. 'Why was retry logic added to fetchUser?').
    """
    result = ask_tool(question)
    return result

def main():
    mcp.run()

if __name__ == "__main__":
    main()
