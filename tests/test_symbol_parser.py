from archaeologist.ingestion.symbol_parser import (
    extract_symbols_from_code,
    map_lines_to_symbols,
    extract_modified_line_numbers_from_diff
)

def test_extract_python_ast_symbols():
    python_code = """
class AuthService:
    def login(self, username, password):
        return True

def fetch_user(user_id):
    return {"id": user_id}
"""
    symbols = extract_symbols_from_code(python_code, "src/auth.py")
    symbol_names = [s["name"] for s in symbols]
    
    assert "AuthService" in symbol_names
    assert "login" in symbol_names
    assert "fetch_user" in symbol_names

def test_map_lines_to_symbols():
    symbols = [
        {"name": "AuthService", "kind": "class", "start_line": 2, "end_line": 5, "symbol_id": "src/auth.py::AuthService"},
        {"name": "fetch_user", "kind": "function", "start_line": 6, "end_line": 8, "symbol_id": "src/auth.py::fetch_user"}
    ]
    
    matched = map_lines_to_symbols(symbols, [3, 4])
    assert matched == ["src/auth.py::AuthService"]

def test_extract_modified_line_numbers_from_diff():
    diff_text = """--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,4 @@
 def login():
+    print("logging in")
     return True
"""
    file_lines = extract_modified_line_numbers_from_diff(diff_text)
    assert "src/auth.py" in file_lines
    assert 11 in file_lines["src/auth.py"]
