import io
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

os.makedirs("docs/images", exist_ok=True)

def make_console(width=96):
    return Console(
        file=io.StringIO(),
        record=True,
        width=width,
        force_terminal=True,
        color_system="truecolor"
    )

# 1. cli_ask.svg
c = make_console()
c.print("[dim]$ archaeologist ask \"Why was retry logic added to fetchUser?\"[/dim]\n")
c.print("[bold cyan]🔍 Forensic Investigation:[/bold cyan] [bold white]Why was retry logic added to fetchUser?[/bold white]\n")

analysis_md = """### Causal Origin
Retry logic with exponential backoff was introduced in commit [`8f31a2c`](#) (merged via PR [#421](#)) by `@tomchristie` to mitigate cascading `503 Upstream Gateway Timeout` failures in payment gateway RPC calls.

### Forensic Chain of Custody
* **Commit `8f31a2c`**: *"Add jittered exponential backoff retry loop to UserService.fetchUser()"*
* **Pull Request #421**: *"Fix transient payment provider drops during peak EU checkout windows"*
* **Linked Issue #389**: Production incident post-mortem: *"Upstream connection timeouts causing 12% failed checkout attempts during flash sale"*
* **AST Scope**: Modified `UserService.fetchUser` and `BaseHTTPClient.execute_with_retry`

### Architectural Impact
The retry policy capped max retries at 3 with a 200ms initial jitter to avoid thundering herd contention on downstream Redis caches."""

c.print(Panel(
    Markdown(analysis_md),
    title="[bold green]🏺 Forensic Analysis & Causal Rationale[/bold green]",
    subtitle="[dim]Grounded in git commits, diffs, AST symbols & PR discussions[/dim]",
    border_style="bright_green",
    box=box.ROUNDED,
    padding=(1, 2)
))
c.save_svg("docs/images/cli_ask.svg", title="archaeologist ask — Forensic Causal Investigation")

# 2. cli_hotspots.svg
c = make_console()
c.print("[dim]$ archaeologist hotspots --top-n 8[/dim]\n")

table = Table(
    title="🔥 Repository Hotspot Files (Commit Frequency)",
    border_style="bright_blue",
    box=box.ROUNDED,
    header_style="bold bright_white on dark_blue"
)
table.add_column("Rank", justify="center", style="bold cyan", no_wrap=True)
table.add_column("File Path", style="bold white")
table.add_column("Commits", justify="right", style="bold magenta")
table.add_column("Churn Visual", justify="left", style="magenta")

hotspots_data = [
    ("#1", "requests/models.py", 717, "████████████████"),
    ("#2", "test_requests.py", 366, "████████"),
    ("#3", "requests/sessions.py", 335, "███████"),
    ("#4", "HISTORY.rst", 319, "███████"),
    ("#5", "requests/utils.py", 271, "██████"),
    ("#6", "tests/test_requests.py", 252, "█████"),
    ("#7", "docs/user/advanced.rst", 225, "█████"),
    ("#8", "docs/index.rst", 187, "████"),
]
for rank, path, count, bar in hotspots_data:
    table.add_row(rank, path, f"{count:,}", f"[magenta]{bar}[/magenta]")

c.print(table)
c.save_svg("docs/images/cli_hotspots.svg", title="archaeologist hotspots — Churn Frequency")

# 3. cli_ownership.svg
c = make_console()
c.print("[dim]$ archaeologist ownership[/dim]\n")

table = Table(
    title="👥 Repository-Wide Author Contribution Distribution",
    border_style="cyan",
    box=box.ROUNDED,
    header_style="bold bright_white on dark_cyan"
)
table.add_column("Author", style="bold white")
table.add_column("Commits", justify="right", style="cyan")
table.add_column("Share %", justify="right", style="bold green")
table.add_column("Contribution", justify="left", style="green")

authors = [
    ("Kenneth Reitz", "3,148", "48.3%", "███████"),
    ("Cory Benfield", "610", "9.4%", "█"),
    ("Nate Prewitt", "351", "5.4%", "█"),
    ("Ian Cordasco", "313", "4.8%", "█"),
    ("Thomas Peterson", "142", "2.2%", "▌"),
    ("Lukasa", "109", "1.7%", "▎"),
]
for a, c_cnt, pct, bar in authors:
    table.add_row(a, c_cnt, pct, f"[green]{bar}[/green]")

c.print(table)
c.print(Panel(
    "[bold red]⚠️  High Bus Factor Risk Detected:[/bold red] Single author accounts for nearly 50% of all changes.\n"
    "[dim]Recommendation: Distribute code reviews and core module ownership to mitigate single-point knowledge loss.[/dim]",
    border_style="bright_red",
    box=box.ROUNDED
))
c.save_svg("docs/images/cli_ownership.svg", title="archaeologist ownership — Bus Factor Risk")

# 4. cli_coupling.svg
c = make_console()
c.print("[dim]$ archaeologist coupling --min-co-commits 50[/dim]\n")

table = Table(
    title="🔗 Temporal Change Coupling (Files Changed Together)",
    border_style="magenta",
    box=box.ROUNDED,
    header_style="bold bright_white on dark_magenta"
)
table.add_column("Rank", justify="center", style="bold cyan", no_wrap=True)
table.add_column("File A", style="bright_white")
table.add_column("File B", style="bright_white")
table.add_column("Co-Commits", justify="right", style="bold magenta")
table.add_column("Coupling Strength", justify="left", style="magenta")

couplings = [
    ("#1", "requests/models.py", "test_requests.py", 281, "████████████"),
    ("#2", "requests/models.py", "requests/sessions.py", 205, "████████"),
    ("#3", "HISTORY.rst", "requests/__init__.py", 166, "███████"),
    ("#4", "requests/models.py", "tests/test_requests.py", 142, "██████"),
    ("#5", "requests/models.py", "requests/utils.py", 114, "█████"),
]
for r, fa, fb, co, bar in couplings:
    table.add_row(r, fa, fb, f"{co:,}", f"[magenta]{bar}[/magenta]")

c.print(table)
c.save_svg("docs/images/cli_coupling.svg", title="archaeologist coupling — Temporal Coupling")

# 5. cli_ingest.svg
c = make_console()
c.print("[dim]$ archaeologist ingest . --window 6m --repo-url https://github.com/psf/requests[/dim]\n")
c.print("[bold cyan]Selected Ingestion Window:[/bold cyan] 6m (Last 6 months: 320 commits)\n")
c.print("[bold cyan]🏺 [ 78%][/bold cyan] [bold green]━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━[/bold green]  [white]78.2%[/white] • [white]Extracting AST symbols & class/method chunks...[/white] [dim]0:00:18[/dim]\n")
c.print("[dim]✓ Parsed 320 git commits & diff hunks[/dim]")
c.print("[dim]✓ Fetched 45 linked GitHub PR reviews & issues[/dim]")
c.print("[dim]✓ Detected 3 bidirectional revert pairs (reverts_sha ↔ superseded_by)[/dim]")
c.print("[dim]✓ Extracted 1,420 AST code symbols (functions, classes, methods)[/dim]")
c.print("[bold green]✓ Ingestion completed in 22.4s. Indexed 1,842 chunks into SQLite + Qdrant.[/bold green]\n")
c.print("You can now run: [bold cyan]archaeologist ask \"Why did ... change?\"[/bold cyan]")
c.save_svg("docs/images/cli_ingest.svg", title="archaeologist ingest — Progress & Indexing")

print("Generated all CLI terminal SVGs successfully.")
