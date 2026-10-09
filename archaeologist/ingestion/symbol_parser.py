import os
import ast
import re
import subprocess
from typing import List, Dict, Any, Optional
from archaeologist.utils.security import validate_repo_path, sanitize_sha, sanitize_file_path

# Require explicit function definition keywords (§4 Fix)
FUNCTION_REGEX = re.compile(
    r'^\s*(?:async\s+)?(?:def|function|fn|func|public|private|protected|static)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE
)
CLASS_REGEX = re.compile(
    r'^\s*(?:class|struct|interface|trait|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
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
    """Parses code text using Python AST if .py, or regex fallback for other languages."""
    if file_path.endswith(".py"):
        try:
            tree = ast.parse(code_text)
            visitor = _SymbolVisitor(file_path)
            visitor.visit(tree)
            return visitor.symbols
        except Exception:
            pass

    symbols = []
    # Multi-language Regex Fallback
    for match in FUNCTION_REGEX.finditer(code_text):
        fn_name = match.group(1)
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
    """Given a list of symbols and modified lines in a diff, returns symbol_ids that overlap."""
    if not symbols or not modified_lines:
        return []

    lines_set = set(modified_lines)
    touched_symbols = []

    sorted_syms = sorted(symbols, key=lambda s: s.get("start_line", s.get("line_number", 0)))
    for i, sym in enumerate(sorted_syms):
        start_line = sym.get("start_line", sym.get("line_number", 0))
        if "end_line" in sym and sym["end_line"] is not None:
            end_line = sym["end_line"]
        else:
            end_line = (
                sorted_syms[i + 1].get("start_line", sorted_syms[i + 1].get("line_number", start_line + 50)) - 1 
                if i + 1 < len(sorted_syms) 
                else start_line + 50
            )
        
        sym_lines = set(range(start_line, end_line + 1))
        if sym_lines.intersection(lines_set):
            if sym["symbol_id"] not in touched_symbols:
                touched_symbols.append(sym["symbol_id"])

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


