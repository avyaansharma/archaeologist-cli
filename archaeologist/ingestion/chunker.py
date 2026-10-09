import os
import ast
import subprocess
import tiktoken
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional


_ENC = None

def get_tiktoken_encoding():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = None
    return _ENC

class _FallbackEncoder:
    def encode(self, text: str) -> list:
        if not text:
            return []
        return [text[i:i+4] for i in range(0, len(text), 4)]

    def decode(self, tokens: list) -> str:
        return "".join(tokens)

class _EncodingProxy:
    def encode(self, text: str) -> list:
        real_enc = get_tiktoken_encoding()
        if real_enc is not None:
            try:
                return real_enc.encode(text)
            except Exception:
                pass
        return _FallbackEncoder().encode(text)

    def decode(self, tokens: list) -> str:
        real_enc = get_tiktoken_encoding()
        if real_enc is not None:
            try:
                return real_enc.decode(tokens)
            except Exception:
                pass
        return _FallbackEncoder().decode(tokens)

enc = _EncodingProxy()

def token_count(text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text))

def make_deterministic_chunk_id(source_type: str, source_id: str, index: int = 0, text_snippet: str = "", repo_id: Optional[str] = None) -> str:
    """Generates a deterministic UUID5 chunk ID based on source metadata, index, and repository scope."""
    key = f"{repo_id or ''}:{source_type}:{source_id}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def chunk_commit(commit: dict, diff_summary: str | None, repo_id: Optional[str] = None) -> List[dict]:
    body = diff_summary or ""
    symbols_str = ", ".join(commit.get("symbols_modified", []))
    sym_header = f"\nSymbols Modified: {symbols_str}" if symbols_str else ""
    raw_msg = commit.get("message") or ""
    actual_repo_id = repo_id or commit.get("repo_id")
    
    # Enrich with causal revert metadata
    revert_meta = ""
    related_ids = list(commit.get("related_ids", []))
    if commit.get("reverts_sha"):
        revert_meta += f"\n[Reverts Commit]: {commit['reverts_sha']}"
        if commit["reverts_sha"] not in related_ids:
            related_ids.append(commit["reverts_sha"])
    if commit.get("superseded_by_sha"):
        revert_meta += f"\n[Superseded by Revert]: {commit['superseded_by_sha']}"
        if commit["superseded_by_sha"] not in related_ids:
            related_ids.append(commit["superseded_by_sha"])
    
    text = f"Commit {commit['sha']} by {commit['author_name']}\nMessage: {raw_msg}{sym_header}{revert_meta}\n\nDiff Summary:\n{body}".strip()
    if token_count(text) > 500:
        msg_prefix = f"Commit {commit['sha']} by {commit['author_name']}\nMessage: "
        msg_suffix = f"{sym_header}{revert_meta}\n\nDiff Summary:\n"
        static_tokens = token_count(msg_prefix) + token_count(msg_suffix)
        max_msg_tokens = max(350 - static_tokens, 100)
        if token_count(raw_msg) > max_msg_tokens:
            raw_msg = enc.decode(enc.encode(raw_msg)[:max_msg_tokens])

        msg_text = f"{msg_prefix}{raw_msg}{msg_suffix}"
        msg_tokens = token_count(msg_text)
        budget = max(500 - msg_tokens, 50)
        body_tokens = enc.encode(body)[:budget]
        body = enc.decode(body_tokens)
        text = f"{msg_text}{body}".strip()
        if token_count(text) > 520:
            text = enc.decode(enc.encode(text)[:500])
        
    is_reverted_commit = bool(commit.get("superseded_by_sha"))
    chunk_id = make_deterministic_chunk_id("commit", commit["sha"], 0, text, repo_id=actual_repo_id)
    chunks = [{
        "id": chunk_id,
        "repo_id": actual_repo_id,
        "source_type": "commit",
        "source_id": commit["sha"],
        "text": text,
        "timestamp": commit["authored_date"],
        "file_paths": commit.get("files_changed", []),
        "symbols_modified": commit.get("symbols_modified", []),
        "is_reverted": is_reverted_commit,
        "token_count": token_count(text),
        "related_ids": related_ids
    }]

    # Handle squash-merge bullet items or long multi-topic commit messages
    bullet_lines = [line.strip() for line in raw_msg.splitlines() if line.strip().startswith(("* ", "- ", "• ")) and len(line.strip()) > 5]
    if bullet_lines:
        for idx, bullet in enumerate(bullet_lines, start=1):
            sub_text = f"Commit {commit['sha']} ({commit.get('author_name', 'unknown')}) sub-topic: {bullet}{sym_header}"
            if token_count(sub_text) > 400:
                sub_text = enc.decode(enc.encode(sub_text)[:350])
            sub_id = make_deterministic_chunk_id("commit", commit["sha"], idx, sub_text, repo_id=actual_repo_id)
            chunks.append({
                "id": sub_id,
                "repo_id": actual_repo_id,
                "source_type": "commit",
                "source_id": commit["sha"],
                "text": sub_text,
                "timestamp": commit["authored_date"],
                "file_paths": commit.get("files_changed", []),
                "symbols_modified": commit.get("symbols_modified", []),
                "is_reverted": is_reverted_commit,
                "token_count": token_count(sub_text),
                "related_ids": list(related_ids)
            })
    elif token_count(raw_msg) > 250:
        paragraphs = [p.strip() for p in raw_msg.split("\n\n") if len(p.strip()) > 20]
        if len(paragraphs) > 1:
            for idx, para in enumerate(paragraphs[1:], start=1):
                sub_text = f"Commit {commit['sha']} note: {para}{sym_header}"
                if token_count(sub_text) > 400:
                    sub_text = enc.decode(enc.encode(sub_text)[:350])
                sub_id = make_deterministic_chunk_id("commit", commit["sha"], idx, sub_text, repo_id=actual_repo_id)
                chunks.append({
                    "id": sub_id,
                    "repo_id": actual_repo_id,
                    "source_type": "commit",
                    "source_id": commit["sha"],
                    "text": sub_text,
                    "timestamp": commit["authored_date"],
                    "file_paths": commit.get("files_changed", []),
                    "symbols_modified": commit.get("symbols_modified", []),
                    "is_reverted": is_reverted_commit,
                    "token_count": token_count(sub_text),
                    "related_ids": list(related_ids)
                })

    return chunks

def chunk_issue(issue: dict, repo_id: Optional[str] = None) -> List[dict]:
    chunks = []
    source_id = f"issue#{issue['number']}"
    author_name = issue.get("author") or "unknown"
    actual_repo_id = repo_id or issue.get("repo_id")
    
    related_ids = [f"pr#{num}" for num in issue.get("linked_pr_numbers", [])]
    for sha in issue.get("linked_commit_shas", []):
        c_ref = f"commit#{sha}"
        if c_ref not in related_ids:
            related_ids.append(c_ref)

    body_text = issue.get("body") or ""
    issue_start = f"[Issue #{issue['number']} - '{issue['title']}' created by {author_name}]\n{body_text}"
    if token_count(issue_start) > 400:
        truncated_body = enc.decode(enc.encode(body_text)[:300])
        issue_start = f"[Issue #{issue['number']} - '{issue['title']}' created by {author_name}]\n{truncated_body}..."

    created_at = issue.get("created_at") or datetime.utcnow()
    
    chunk_id_head = make_deterministic_chunk_id("issue", source_id, 0, issue_start, repo_id=actual_repo_id)
    chunks.append({
        "id": chunk_id_head,
        "repo_id": actual_repo_id,
        "source_type": "issue",
        "source_id": source_id,
        "text": issue_start,
        "timestamp": created_at,
        "file_paths": [],
        "symbols_modified": [],
        "is_reverted": False,
        "token_count": token_count(issue_start),
        "related_ids": related_ids
    })

    for idx, c in enumerate(issue.get("comments", []), start=1):
        c_author = c.get("author") or "unknown"
        c_body = c.get("body") or ""
        comment_text = f"[Issue #{issue['number']} comment by {c_author}]: {c_body}"
        if token_count(comment_text) > 400:
            truncated = enc.decode(enc.encode(c_body)[:350])
            comment_text = f"[Issue #{issue['number']} comment by {c_author}]: {truncated}..."
        
        c_time = c.get("created_at")
        if isinstance(c_time, str):
            try:
                c_time = datetime.fromisoformat(c_time)
            except Exception:
                c_time = created_at
        else:
            c_time = c_time or created_at
            
        chunk_id_comm = make_deterministic_chunk_id("issue", source_id, idx, comment_text, repo_id=actual_repo_id)
        chunks.append({
            "id": chunk_id_comm,
            "repo_id": actual_repo_id,
            "source_type": "issue",
            "source_id": source_id,
            "text": comment_text,
            "timestamp": c_time,
            "file_paths": [],
            "symbols_modified": [],
            "is_reverted": False,
            "token_count": token_count(comment_text),
            "related_ids": related_ids
        })

    return chunks

def chunk_pr(pr: dict, repo_id: Optional[str] = None) -> List[dict]:
    chunks = []
    source_id = f"pr#{pr['number']}"
    author_name = pr.get("author") or "unknown"
    actual_repo_id = repo_id or pr.get("repo_id")
    
    related_ids = [f"issue#{num}" for num in pr.get("linked_issue_numbers", [])]
    merge_sha = pr.get("merge_commit_sha") or pr.get("merged_commit_sha")
    if merge_sha:
        m_ref = f"commit#{merge_sha}"
        if m_ref not in related_ids:
            related_ids.append(m_ref)
    for sha in pr.get("linked_commit_shas", []):
        c_ref = f"commit#{sha}"
        if c_ref not in related_ids:
            related_ids.append(c_ref)

    body_text = pr.get("body") or ""
    pr_start = f"[PR #{pr['number']} - '{pr['title']}' created by {author_name}]\n{body_text}"
    if token_count(pr_start) > 400:
        truncated_body = enc.decode(enc.encode(body_text)[:300])
        pr_start = f"[PR #{pr['number']} - '{pr['title']}' created by {author_name}]\n{truncated_body}..."

    created_at = pr.get("created_at") or datetime.utcnow()
    
    chunk_id_head = make_deterministic_chunk_id("pr", source_id, 0, pr_start, repo_id=actual_repo_id)
    chunks.append({
        "id": chunk_id_head,
        "repo_id": actual_repo_id,
        "source_type": "pr",
        "source_id": source_id,
        "text": pr_start,
        "timestamp": created_at,
        "file_paths": [],
        "symbols_modified": [],
        "is_reverted": False,
        "token_count": token_count(pr_start),
        "related_ids": related_ids
    })

    # Process genuine inline review comments on diff lines
    idx_count = 1
    for c in pr.get("review_comments", []):
        c_author = c.get("author") or "unknown"
        c_body = c.get("body") or ""
        f_path = c.get("path", "")
        line_num = c.get("line", "")
        path_info = f" on file '{f_path}' line {line_num}" if f_path else ""
        comment_text = f"[PR #{pr['number']} inline review comment by {c_author}{path_info}]: {c_body}"
        if token_count(comment_text) > 400:
            truncated = enc.decode(enc.encode(c_body)[:350])
            comment_text = f"[PR #{pr['number']} inline review comment by {c_author}{path_info}]: {truncated}..."
            
        c_time = c.get("created_at")
        if isinstance(c_time, str):
            try:
                c_time = datetime.fromisoformat(c_time)
            except Exception:
                c_time = created_at
        else:
            c_time = c_time or created_at

        chunk_id_comm = make_deterministic_chunk_id("pr", source_id, idx_count, comment_text, repo_id=actual_repo_id)
        chunks.append({
            "id": chunk_id_comm,
            "repo_id": actual_repo_id,
            "source_type": "pr",
            "source_id": source_id,
            "text": comment_text,
            "timestamp": c_time,
            "file_paths": [f_path] if f_path else [],
            "symbols_modified": [],
            "is_reverted": False,
            "token_count": token_count(comment_text),
            "related_ids": related_ids
        })
        idx_count += 1

    # Process general PR discussion comments
    for c in pr.get("comments", []):
        c_author = c.get("author") or "unknown"
        c_body = c.get("body") or ""
        comment_text = f"[PR #{pr['number']} comment by {c_author}]: {c_body}"
        if token_count(comment_text) > 400:
            truncated = enc.decode(enc.encode(c_body)[:350])
            comment_text = f"[PR #{pr['number']} comment by {c_author}]: {truncated}..."
            
        c_time = c.get("created_at")
        if isinstance(c_time, str):
            try:
                c_time = datetime.fromisoformat(c_time)
            except Exception:
                c_time = created_at
        else:
            c_time = c_time or created_at

        chunk_id_comm = make_deterministic_chunk_id("pr", source_id, idx_count, comment_text, repo_id=actual_repo_id)
        chunks.append({
            "id": chunk_id_comm,
            "repo_id": actual_repo_id,
            "source_type": "pr",
            "source_id": source_id,
            "text": comment_text,
            "timestamp": c_time,
            "file_paths": [],
            "symbols_modified": [],
            "is_reverted": False,
            "token_count": token_count(comment_text),
            "related_ids": related_ids
        })
        idx_count += 1

    return chunks


# Minimum token threshold for a method to get its own individual chunk during class decomposition.
# Methods below this threshold are grouped into the class header chunk to avoid tiny-chunk spam.
_MIN_METHOD_TOKENS = 20


def _build_class_header(node: ast.ClassDef, lines: List[str], rel_path: str) -> str:
    """Builds a class header string: class signature + docstring + attribute annotations.
    
    Does NOT include method bodies — those are decomposed into separate chunks.
    """
    # Start with the class definition line(s) and any decorators
    header_parts = []
    
    # Class definition line
    class_line = lines[node.lineno - 1]
    header_parts.append(class_line)
    
    # Extract docstring if present
    if node.body and isinstance(node.body[0], ast.Expr):
        val = node.body[0].value
        ast_str = getattr(ast, "Str", None)
        is_str_constant = isinstance(val, ast.Constant) and isinstance(val.value, str)
        if is_str_constant or (ast_str and isinstance(val, ast_str)):
            doc_node = node.body[0]
            doc_end = getattr(doc_node, 'end_lineno', doc_node.lineno)
            doc_lines = lines[doc_node.lineno - 1 : doc_end]
            header_parts.extend(doc_lines)
    
    # Extract class-level attribute annotations and assignments (not methods)
    for child in node.body:
        if isinstance(child, (ast.AnnAssign, ast.Assign)):
            child_end = getattr(child, 'end_lineno', child.lineno)
            attr_lines = lines[child.lineno - 1 : child_end]
            header_parts.extend(attr_lines)
    
    return "\n".join(header_parts)


def chunk_codebase(repo_path: str, repo_id: Optional[str] = None) -> List[dict]:
    """Walks the target codebase repository, parses AST symbols in current Python files,
    and creates baseline code chunks.
    
    For classes exceeding 600 tokens, applies method-level decomposition:
      1. A class header chunk (signature + docstring + __init__) with symbols_modified
         containing the class name and all tiny method names grouped into it.
      2. Individual per-method chunks for each method >= _MIN_METHOD_TOKENS, each with
         symbols_modified = [ClassName, method_name].
    
    For classes fitting within 600 tokens, creates a single chunk with all method names
    in symbols_modified (safe because the full code is physically present).
    
    For top-level functions, creates a single chunk per function (unchanged behavior).
    """
    chunks = []
    py_files = []

    # Attempt to list git-tracked files first to avoid untracked secrets / logs
    git_dir = os.path.join(repo_path, ".git")
    if os.path.exists(git_dir):
        try:
            res = subprocess.run(
                ["git", "-C", repo_path, "ls-files", "*.py"],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if res.returncode == 0 and res.stdout.strip():
                for line in res.stdout.splitlines():
                    clean_rel = line.strip().replace("\\", "/")
                    if clean_rel and not any(part in ("venv", ".venv", "tests", "__pycache__", "build", "dist") for part in clean_rel.split("/")):
                        abs_p = os.path.join(repo_path, clean_rel)
                        if os.path.isfile(abs_p):
                            py_files.append((abs_p, clean_rel))
        except Exception:
            py_files = []

    # Fallback to os.walk if not a git repository or git ls-files returned empty
    if not py_files:
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", ".venv", "tests", "__pycache__", "build", "dist", "node_modules", "scratch")]
            for file in files:
                if not file.endswith(".py"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path).replace("\\", "/")
                py_files.append((full_path, rel_path))

    for full_path, rel_path in py_files:
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                if not content.strip():
                    continue

                top_nodes = []
                try:
                    tree = ast.parse(content)
                    top_nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                except Exception:
                    pass

                file_lines = content.splitlines()
                if top_nodes:
                    for idx, node in enumerate(top_nodes):
                        end_ln = getattr(node, 'end_lineno', node.lineno + 60)
                        node_lines = file_lines[node.lineno - 1 : end_ln]
                        node_code = "\n".join(node_lines)
                        node_tokens = token_count(node_code)

                        # --- ClassDef that exceeds 600 tokens: decompose into method-level chunks ---
                        if isinstance(node, ast.ClassDef) and node_tokens > 600:
                            inner_methods = [
                                sub for sub in node.body
                                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            ]
                            
                            # 1. Build class header chunk (signature + docstring + attrs + tiny methods)
                            header_text = _build_class_header(node, file_lines, rel_path)
                            
                            # Collect tiny methods to group into the header chunk
                            tiny_method_names = []
                            for m in inner_methods:
                                m_end = getattr(m, 'end_lineno', m.lineno + 30)
                                m_code = "\n".join(file_lines[m.lineno - 1 : m_end])
                                m_toks = token_count(m_code)
                                if m_toks < _MIN_METHOD_TOKENS:
                                    tiny_method_names.append(m.name)
                                    header_text += "\n" + m_code
                            
                            # Include __init__ in the header chunk (it provides class context)
                            for m in inner_methods:
                                if m.name == "__init__" and m.name not in tiny_method_names:
                                    m_end = getattr(m, 'end_lineno', m.lineno + 30)
                                    m_code = "\n".join(file_lines[m.lineno - 1 : m_end])
                                    m_toks = token_count(m_code)
                                    if m_toks <= 500:
                                        header_text += "\n" + m_code
                                    else:
                                        header_text += "\n" + enc.decode(enc.encode(m_code)[:400])
                                    break
                            
                            header_chunk_text = f"File: {rel_path}\nClass: {node.name}\n\n{header_text}".strip()
                            if token_count(header_chunk_text) > 600:
                                header_chunk_text = enc.decode(enc.encode(header_chunk_text)[:500])
                            
                            header_symbols = [node.name] + tiny_method_names
                            header_id = make_deterministic_chunk_id("code", f"{rel_path}:{node.name}", idx, header_chunk_text, repo_id=repo_id)
                            chunks.append({
                                "id": header_id,
                                "repo_id": repo_id,
                                "source_type": "code",
                                "source_id": rel_path,
                                "text": header_chunk_text,
                                "timestamp": datetime.utcnow(),
                                "file_paths": [rel_path],
                                "symbols_modified": list(dict.fromkeys(header_symbols)),
                                "is_reverted": False,
                                "token_count": token_count(header_chunk_text),
                                "related_ids": []
                            })
                            
                            # 2. Create individual per-method chunks for substantial methods
                            method_sub_idx = 0
                            for m in inner_methods:
                                if m.name in tiny_method_names or m.name == "__init__":
                                    continue
                                 
                                m_end = getattr(m, 'end_lineno', m.lineno + 30)
                                m_code = "\n".join(file_lines[m.lineno - 1 : m_end])
                                m_toks = token_count(m_code)
                                 
                                if m_toks > 600:
                                    m_code = enc.decode(enc.encode(m_code)[:500])
                                 
                                method_chunk_text = f"File: {rel_path}\nClass: {node.name}\nMethod: {m.name}\n\n{m_code}".strip()
                                method_sub_idx += 1
                                method_id = make_deterministic_chunk_id(
                                    "code", f"{rel_path}:{node.name}.{m.name}", method_sub_idx, method_chunk_text, repo_id=repo_id
                                )
                                chunks.append({
                                    "id": method_id,
                                    "repo_id": repo_id,
                                    "source_type": "code",
                                    "source_id": rel_path,
                                    "text": method_chunk_text,
                                    "timestamp": datetime.utcnow(),
                                    "file_paths": [rel_path],
                                    "symbols_modified": [node.name, m.name],
                                    "is_reverted": False,
                                    "token_count": token_count(method_chunk_text),
                                    "related_ids": []
                                })
                         
                        else:
                            # --- Small class or top-level function: single chunk ---
                            if node_tokens > 600:
                                node_code = enc.decode(enc.encode(node_code)[:500])
                             
                            # For small classes, enrich symbols_modified with method names
                            chunk_symbols = [node.name]
                            if isinstance(node, ast.ClassDef):
                                for sub in node.body:
                                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                        if sub.name not in chunk_symbols:
                                            chunk_symbols.append(sub.name)
                             
                            chunk_text = f"File: {rel_path}\nSymbol: {node.name}\n\n{node_code}".strip()
                            chunk_id = make_deterministic_chunk_id("code", f"{rel_path}:{node.name}", idx, chunk_text, repo_id=repo_id)
                            chunks.append({
                                "id": chunk_id,
                                "repo_id": repo_id,
                                "source_type": "code",
                                "source_id": rel_path,
                                "text": chunk_text,
                                "timestamp": datetime.utcnow(),
                                "file_paths": [rel_path],
                                "symbols_modified": list(dict.fromkeys(chunk_symbols)),
                                "is_reverted": False,
                                "token_count": token_count(chunk_text),
                                "related_ids": []
                            })
                else:
                    symbols_found = []
                    chunk_text = f"File: {rel_path}\n\n{content[:1500]}".strip()
                    chunk_id = make_deterministic_chunk_id("code", rel_path, 0, chunk_text, repo_id=repo_id)
                    chunks.append({
                        "id": chunk_id,
                        "repo_id": repo_id,
                        "source_type": "code",
                        "source_id": rel_path,
                        "text": chunk_text,
                        "timestamp": datetime.utcnow(),
                        "file_paths": [rel_path],
                        "symbols_modified": symbols_found,
                        "is_reverted": False,
                        "token_count": token_count(chunk_text),
                        "related_ids": []
                    })
            except Exception:
                pass
    return chunks



