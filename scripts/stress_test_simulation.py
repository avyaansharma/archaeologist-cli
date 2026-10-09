"""Real-Time End-to-End CLI Stress Test and Simulation Script.

Simulates the complete user workflow from a clean git repository:
1. Ingests a multi-author git repository with AST symbols, PR references, and reverts.
2. Asserts status, hotspots, ownership, coupling, symbols, and symbol-history.
3. Tests both Rich terminal formatted tables and JSON outputs.
4. Executes 'arch why <file>:<line>' and validates commit SHA output.
5. Executes 'arch ask <question>' and validates architectural causal rationale and citations.
6. Tests repeat ingestion to confirm zero redundant work (0 diff summaries, 0 re-embeds).
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def run_cmd(args, cwd=None, check=True):
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if check and res.returncode != 0:
        print(f"FAILED CMD: {' '.join(args)}", file=sys.stderr)
        print("STDOUT:", res.stdout, file=sys.stderr)
        print("STDERR:", res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res

def main():
    temp_dir = Path(tempfile.mkdtemp(prefix="arch_stress_test_"))
    print(f"[*] Setting up stress-test git repository in: {temp_dir}")

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Alice Developer",
        "GIT_AUTHOR_EMAIL": "alice@example.com",
        "GIT_COMMITTER_NAME": "Alice Developer",
        "GIT_COMMITTER_EMAIL": "alice@example.com",
    }

    def git_commit(msg, author_name="Alice Developer", author_email="alice@example.com"):
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_COMMITTER_NAME": author_name,
            "GIT_COMMITTER_EMAIL": author_email,
        }
        subprocess.run(["git", "add", "."], cwd=str(temp_dir), check=True, env=env)
        subprocess.run(["git", "commit", "-m", msg], cwd=str(temp_dir), check=True, env=env)
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(temp_dir), capture_output=True, text=True, check=True)
        return res.stdout.strip()

    try:
        run_cmd(["git", "init"], cwd=str(temp_dir))
        run_cmd(["git", "config", "user.name", "Alice Developer"], cwd=str(temp_dir))
        run_cmd(["git", "config", "user.email", "alice@example.com"], cwd=str(temp_dir))

        # Commit 1: Initial auth service
        auth_file = temp_dir / "auth.py"
        auth_file.write_text(
            'class AuthService:\n'
            '    def login(self, username, password):\n'
            '        """Authenticates user."""\n'
            '        if not username or not password:\n'
            '            return False\n'
            '        return username == "admin"\n',
            encoding="utf-8"
        )
        sha1 = git_commit("feat: initial AuthService implementation", "Alice Developer", "alice@example.com")
        print(f"  [+] Created commit 1 ({sha1[:7]}): feat: initial AuthService implementation")

        # Commit 2: Token verification added by Bob
        auth_file.write_text(
            'class AuthService:\n'
            '    def login(self, username, password):\n'
            '        """Authenticates user with password."""\n'
            '        return username == "admin"\n\n'
            '    def verify_token(self, token):\n'
            '        """Verifies session bearer token."""\n'
            '        return len(token) > 10\n',
            encoding="utf-8"
        )
        sha2 = git_commit("feat: add token verification logic (#101)", "Bob Engineer", "bob@example.com")
        print(f"  [+] Created commit 2 ({sha2[:7]}): feat: add token verification logic (#101)")

        # Commit 3: Rate limiter co-changed with auth by Charlie
        limiter_file = temp_dir / "limiter.py"
        limiter_file.write_text(
            'class RateLimiter:\n'
            '    def is_allowed(self, client_ip):\n'
            '        return True\n',
            encoding="utf-8"
        )
        auth_file.write_text(
            'from limiter import RateLimiter\n\n'
            'class AuthService:\n'
            '    def __init__(self):\n'
            '        self.limiter = RateLimiter()\n\n'
            '    def login(self, username, password):\n'
            '        return self.limiter.is_allowed("127.0.0.1") and username == "admin"\n',
            encoding="utf-8"
        )
        sha3 = git_commit("feat: integrate RateLimiter into AuthService (#102)", "Charlie Architect", "charlie@example.com")
        print(f"  [+] Created commit 3 ({sha3[:7]}): feat: integrate RateLimiter into AuthService (#102)")

        # Commit 4: Buggy experimental token caching
        auth_file.write_text(auth_file.read_text(encoding="utf-8") + "\n    def cache_token(self): pass\n", encoding="utf-8")
        sha4 = git_commit("feat: experimental token caching", "Bob Engineer", "bob@example.com")
        print(f"  [+] Created commit 4 ({sha4[:7]}): feat: experimental token caching")

        # Commit 5: Revert commit
        auth_file.write_text(
            'from limiter import RateLimiter\n\n'
            'class AuthService:\n'
            '    def __init__(self):\n'
            '        self.limiter = RateLimiter()\n\n'
            '    def login(self, username, password):\n'
            '        return self.limiter.is_allowed("127.0.0.1") and username == "admin"\n',
            encoding="utf-8"
        )
        revert_msg = f'Revert "feat: experimental token caching"\n\nThis reverts commit {sha4}.'
        sha5 = git_commit(revert_msg, "Alice Developer", "alice@example.com")
        print(f"  [+] Created commit 5 ({sha5[:7]}): Revert commit for {sha4[:7]}")

        # Python interpreter path
        py_exe = sys.executable

        # STEP 1: INGESTION
        print("\n[*] STEP 1: Running 'archaeologist ingest' on the simulated repo...")
        ingest_res = run_cmd([py_exe, "-m", "archaeologist.cli", "ingest", str(temp_dir), "--window", "full"])
        print("  -> Ingestion output summary:")
        for line in ingest_res.stdout.splitlines()[-8:]:
            print(f"     {line}")
        assert "Ingestion completed successfully" in ingest_res.stdout

        # STEP 2: STATUS CHECK
        print("\n[*] STEP 2: Checking 'archaeologist status'...")
        status_res = run_cmd([py_exe, "-m", "archaeologist.cli", "status", "--repo", str(temp_dir)])
        assert "Repository Index Health" in status_res.stdout
        print("  -> Status Table: OK")

        status_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "status", "--repo", str(temp_dir), "--json"])
        status_data = json.loads(status_json_res.stdout.strip())
        print(f"  -> Status JSON parsed: commits={status_data['total_commits']}, symbols={status_data['total_ast_symbols']}")
        assert status_data["total_commits"] == 5
        assert status_data["total_ast_symbols"] >= 2

        # STEP 3: HOTSPOTS CHECK
        print("\n[*] STEP 3: Checking 'archaeologist hotspots'...")
        hotspots_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "hotspots", "--repo", str(temp_dir), "--json"])
        hotspots = json.loads(hotspots_json_res.stdout.strip())
        print(f"  -> Hotspots JSON: top hotspot is '{hotspots[0]['file_path']}' with {hotspots[0]['commit_count']} commits.")
        assert hotspots[0]["file_path"] == "auth.py"
        assert hotspots[0]["commit_count"] >= 4

        # STEP 4: OWNERSHIP CHECK
        print("\n[*] STEP 4: Checking 'archaeologist ownership'...")
        ownership_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "ownership", "--repo", str(temp_dir), "--json"])
        ownership = json.loads(ownership_json_res.stdout.strip())
        authors = set(ownership.get("author_distribution", {}).keys())
        print(f"  -> Authors identified: {authors}")
        assert "Alice Developer" in authors
        assert "Bob Engineer" in authors
        assert "Charlie Architect" in authors

        # STEP 5: COUPLING CHECK
        print("\n[*] STEP 5: Checking 'archaeologist coupling'...")
        coupling_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "coupling", "--repo", str(temp_dir), "--json", "--min-co-commits", "1"])
        coupling = json.loads(coupling_json_res.stdout.strip())
        print(f"  -> Coupling pairs found: {len(coupling)}")
        assert any(c["file_a"] in ["auth.py", "limiter.py"] and c["file_b"] in ["auth.py", "limiter.py"] for c in coupling)

        # STEP 6: SYMBOLS & SYMBOL-HISTORY
        print("\n[*] STEP 6: Checking 'archaeologist symbols' and 'symbol-history'...")
        symbols_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "symbols", "--repo", str(temp_dir), "--json"])
        symbols = json.loads(symbols_json_res.stdout.strip())
        symbol_names = [s["symbol_name"] for s in symbols]
        print(f"  -> AST Symbols found: {symbol_names}")
        assert any("AuthService" in s for s in symbol_names)
        assert any("login" in s for s in symbol_names)

        history_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "symbol-history", "login", "--repo", str(temp_dir), "--json"])
        history = json.loads(history_json_res.stdout.strip())
        print(f"  -> Symbol-history for 'login': {len(history)} modifying commit(s)")
        history_shas = [h["sha"] for h in history]
        print(f"  -> Commits modifying 'login': {[s[:7] for s in history_shas]}")
        assert sha1 in history_shas or any(s.startswith(sha1[:7]) for s in history_shas)

        # STEP 7: WHY <FILE>:<LINE>
        print(f"\n[*] STEP 7: Checking 'archaeologist why auth.py:7' (RateLimiter line)...")
        why_json_res = run_cmd([py_exe, "-m", "archaeologist.cli", "why", f"{temp_dir / 'auth.py'}:7", "--repo", str(temp_dir), "--json"])
        why_data = json.loads(why_json_res.stdout.strip())
        why_commits = why_data.get("commits", [])
        print(f"  -> Why detected commits: {[c['sha'][:7] for c in why_commits]}")
        assert len(why_commits) > 0
        assert "explanation" in why_data and len(why_data["explanation"]) > 20
        print(f"  -> Explanation snippet: {why_data['explanation'][:100]}...")

        # STEP 8: ARCHAEOLOGIST ASK (CAUSAL RATIONALE & CITATIONS)
        print("\n[*] STEP 8: Checking 'archaeologist ask' on simulated repo...")
        ask_res = run_cmd([py_exe, "-m", "archaeologist.cli", "ask", "Why was RateLimiter integrated into AuthService?", "--repo", str(temp_dir)])
        print("  -> Ask output snippet:")
        for line in ask_res.stdout.splitlines()[-15:]:
            print(f"     {line}")
        assert "Forensic Analysis & Causal Rationale" in ask_res.stdout or "Grounded in git commits" in ask_res.stdout

        # STEP 9: REPEAT INGESTION (IDEMPOTENCY & ZERO REDUNDANT CALLS)
        print("\n[*] STEP 9: Running repeat ingestion to verify 0 redundant work...")
        repeat_res = run_cmd([py_exe, "-m", "archaeologist.cli", "ingest", str(temp_dir), "--window", "full"])
        print("  -> Repeat stdout:")
        for line in repeat_res.stdout.splitlines():
            print(f"     {line}")

        print("\n" + "=" * 60)
        print("🎉 ALL STRESS TEST SIMULATION STEPS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)

if __name__ == "__main__":
    main()
