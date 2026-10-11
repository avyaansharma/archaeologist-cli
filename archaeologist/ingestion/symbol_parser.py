import os
import ast
import re
import subprocess
from typing import List, Dict, Any, Optional
from archaeologist.utils.security import validate_repo_path, sanitize_sha, sanitize_file_path

# Multi-language regex patterns supporting Python, JS/TS, Go, Java/C#, and Rust (§4 Fix)
FUNCTION_REGEX = re.compile(
    r'^\s*(?:@\w+(?:\([^)]*\))?\s*)*'
    r'(?:export\s+(?:default\s+)?)?'
    r'(?:(?:public|private|protected|static|final|native|synchronized|abstract|async|pub)\s+)*'
    r'(?:def|function|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|'
    r'^\s*(?:func)\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|'
    r'^\s*(?:(?:public|private|protected|static|final|native|synchronized|abstract)\s+)+'
    r'(?:[a-zA-Z0-9_<>\[\],\s]+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE
)
CLASS_REGEX = re.compile(
    r'^\s*(?:export\s+(?:default\s+)?)?(?:pub\s+)?(?:public\s+|abstract\s+|final\s+)*'
    r'(?:class|struct|interface|trait|enum|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    re.MULTILINE
)

def resolve_file_path(repo_path: str, fpath: str) -> Optional[str]:
    """Resolves relative git diff path against local repo root directory."""
    p1 = os.path.join(repo_path, fpath)
    if os.path.exists(p1) and os.path.isfile(p1):
        return p1
    parts = fpath.replace("\\", "/").split("/")
    for i in range(len(parts)):
        sub_path = os.path.join(repo_path, *parts[i:])
        if os.path.exists(sub_path) and os.path.isfile(sub_path):
            return sub_path
    return None

def get_git_file_content(repo_path: str, sha: str, file_path: str) -> Optional[str]:
    """Retrieves file content at a specific historical git commit SHA using `git show sha:file_path`."""
    try:
        validated_repo = validate_repo_path(repo_path)
        clean_sha = sanitize_sha(sha)
        clean_path = sanitize_file_path(validated_repo, file_path)
        cmd = ["git", "-C", validated_repo, "show", f"{clean_sha}:{clean_path}"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None

class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.class_stack: List[str] = []
        self.symbols: List[Dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        qualified = "::".join(self.class_stack + [node.name]) if self.class_stack else node.name
        self.symbols.append({
            "symbol_id": f"{self.file_path}::{qualified}",
            "name": node.name,
            "kind": "class",
            "line_number": node.lineno,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 50)
        })
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._record_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._record_func(node)

    def _record_func(self, node: Any):
        if self.class_stack:
            qualified = "::".join(self.class_stack + [node.name])
            sym_id = f"{self.file_path}::{qualified}"
            kind = "method"
        else:
            sym_id = f"{self.file_path}::{node.name}"
            kind = "function"
        self.symbols.append({
            "symbol_id": sym_id,
            "name": node.name,
            "kind": kind,
            "line_number": node.lineno,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 20)
        })
        self.generic_visit(node)

def extract_symbols_from_code(code_text: str, file_path: str) -> List[Dict[str, Any]]:
    """Parses code text using Python AST if .py, or multi-language regex fallback for other languages."""
    if file_path.endswith(".py"):
        try:
            tree = ast.parse(code_text)
            visitor = _SymbolVisitor(file_path)
            visitor.visit(tree)
            return visitor.symbols
        except Exception:
            pass

    symbols: List[Dict[str, Any]] = []
    lines = code_text.splitlines()
    is_python = file_path.endswith(".py")

    # Class scope stack:
    # Python: list of (indent_level, class_name)
    # Curly-brace languages: list of (brace_depth, class_name)
    class_scope_stack: List[tuple] = []
    current_brace_depth = 0

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue

        indent_len = len(line) - len(line.lstrip())

        if not is_python:
            open_braces = line.count("{")
            close_braces = line.count("}")
        else:
            open_braces = close_braces = 0

        # Pop finished scopes
        if is_python:
            while class_scope_stack and indent_len <= class_scope_stack[-1][0]:
                class_scope_stack.pop()
        else:
            while class_scope_stack and current_brace_depth <= class_scope_stack[-1][0]:
                class_scope_stack.pop()

        # Check for class match
        cls_match = CLASS_REGEX.match(line)
        if cls_match:
            cls_name = cls_match.group(1)
            qualified_cls = "::".join([c[1] for c in class_scope_stack] + [cls_name]) if class_scope_stack else cls_name
            symbols.append({
                "symbol_id": f"{file_path}::{qualified_cls}",
                "name": cls_name,
                "kind": "class",
                "line_number": line_idx,
                "start_line": line_idx
            })
            if is_python:
                class_scope_stack.append((indent_len, cls_name))
            else:
                target_depth = current_brace_depth + (1 if "{" in line else 0)
                class_scope_stack.append((target_depth, cls_name))

        # Check for function/method match
        fn_match = FUNCTION_REGEX.match(line)
        if fn_match:
            fn_name = fn_match.group(1) or fn_match.group(2) or fn_match.group(3)
            if fn_name:
                if class_scope_stack:
                    qualified_fn = "::".join([c[1] for c in class_scope_stack] + [fn_name])
                    kind = "method"
                else:
                    qualified_fn = fn_name
                    kind = "function"
                symbols.append({
                    "symbol_id": f"{file_path}::{qualified_fn}",
                    "name": fn_name,
                    "kind": kind,
                    "line_number": line_idx,
                    "start_line": line_idx
                })

        if not is_python:
            current_brace_depth += open_braces - close_braces

    # Global regex fallback if line scan found no symbols
    if not symbols:
        for match in FUNCTION_REGEX.finditer(code_text):
            fn_name = match.group(1) or match.group(2) or match.group(3)
            if not fn_name:
                continue
            line_no = code_text[:match.start()].count("\n") + 1
            symbols.append({
                "symbol_id": f"{file_path}::{fn_name}",
                "name": fn_name,
                "kind": "function",
                "line_number": line_no,
                "start_line": line_no
            })

        for match in CLASS_REGEX.finditer(code_text):
            cls_name = match.group(1)
            line_no = code_text[:match.start()].count("\n") + 1
            symbols.append({
                "symbol_id": f"{file_path}::{cls_name}",
                "name": cls_name,
                "kind": "class",
                "line_number": line_no,
                "start_line": line_no
            })

    return symbols


def map_lines_to_symbols(symbols: List[Dict[str, Any]], modified_lines: List[int]) -> List[str]:
    """Given a list of symbols and modified lines in a diff, returns symbol_ids that overlap.
    Prioritizes the most specific inner symbol (smallest line range).
    Enclosing classes are only attributed if modified lines fall outside any enclosed method.
    """
    if not symbols or not modified_lines:
        return []

    lines_set = set(modified_lines)
    symbol_spans = []
    sorted_syms = sorted(symbols, key=lambda s: s.get("start_line", s.get("line_number", 0)))
    for i, sym in enumerate(sorted_syms):
        s_line = sym.get("start_line", sym.get("line_number", 0))
        if "end_line" in sym and sym["end_line"] is not None:
            e_line = sym["end_line"]
        else:
            e_line = (
                sorted_syms[i + 1].get("start_line", sorted_syms[i + 1].get("line_number", s_line + 50)) - 1
                if i + 1 < len(sorted_syms)
                else s_line + 50
            )
        span_length = max(1, e_line - s_line + 1)
        symbol_spans.append((sym, s_line, e_line, span_length))

    touched_symbols = []
    for line in sorted(list(lines_set)):
        matching = [
            (sym, span_len)
            for (sym, s_line, e_line, span_len) in symbol_spans
            if s_line <= line <= e_line
        ]
        if matching:
            # Most specific symbol has the smallest line span
            matching.sort(key=lambda item: item[1])
            best_sym = matching[0][0]
            sym_id = best_sym["symbol_id"]
            if sym_id not in touched_symbols:
                touched_symbols.append(sym_id)

    return touched_symbols

class DiffLineBucket(dict):
    """Dictionary holding 'added' and 'deleted' line numbers, with backward-compatible list operations."""
    def __init__(self, added: Optional[List[int]] = None, deleted: Optional[List[int]] = None):
        super().__init__({"added": added or [], "deleted": deleted or []})

    def __contains__(self, item):
        if isinstance(item, str):
            return super().__contains__(item)
        return item in self["added"] or item in self["deleted"]

    def __iter__(self):
        combined = set(self["added"]) | set(self["deleted"])
        return iter(sorted(list(combined)))

def extract_modified_line_numbers_from_diff(diff_text: str) -> Dict[str, DiffLineBucket]:
    """Parses a unified git diff and returns a dict mapping file paths to DiffLineBucket (added and deleted line numbers)."""
    file_lines = {}
    current_file = None
    candidate_old_file = None
    old_line = 0
    new_line = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            in_hunk = False
            candidate_old_file = None
        elif re.match(r'^---\s+(a/|/dev/null|"a/|\'a/)', line):
            in_hunk = False
            target = line[4:].strip().strip('"\'')
            candidate_old_file = target[2:] if target.startswith('a/') else None
        elif not in_hunk and line.startswith('--- '):
            target = line[4:].strip().strip('"\'')
            candidate_old_file = target[2:] if target.startswith('a/') else None
        elif re.match(r'^\+\+\+\s+(b/|/dev/null|"b/|\'b/)', line):
            in_hunk = False
            target = line[4:].strip().strip('"\'')
            if target == '/dev/null' or target == 'dev/null':
                current_file = candidate_old_file
            else:
                current_file = target[2:] if target.startswith('b/') else candidate_old_file
            if current_file and current_file not in file_lines:
                file_lines[current_file] = DiffLineBucket([], [])
        elif not in_hunk and line.startswith('+++ '):
            target = line[4:].strip().strip('"\'')
            if target == '/dev/null' or target == 'dev/null':
                current_file = candidate_old_file
            else:
                current_file = target[2:] if target.startswith('b/') else candidate_old_file
            if current_file and current_file not in file_lines:
                file_lines[current_file] = DiffLineBucket([], [])
        elif line.startswith('@@'):
            in_hunk = True
            m_old = re.search(r'-(\d+)', line)
            m_new = re.search(r'\+(\d+)', line)
            if m_old:
                old_line = int(m_old.group(1))
            if m_new:
                new_line = int(m_new.group(1))
        elif current_file and in_hunk:
            if line.startswith('+'):
                file_lines[current_file]["added"].append(new_line)
                new_line += 1
            elif line.startswith('-'):
                file_lines[current_file]["deleted"].append(old_line)
                old_line += 1
            elif not line.startswith('\\'):
                old_line += 1
                new_line += 1

    return file_lines


