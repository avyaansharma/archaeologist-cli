import typer
import os
import json
from typing import Optional
from dotenv import load_dotenv

from archaeologist.storage.paths import (
    find_repo_root,
    get_default_db_url,
    get_default_db_path,
    get_default_bm25_path,
    detect_github_remote,
    calculate_window_since,
    resolve_repo_id,
    get_stored_repo_id
)
from archaeologist.utils.config import sync_env_from_config, load_user_config, save_user_config

def _configure_repo_env(repo_path: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    """Sets database, Qdrant, repo, and BM25 environment variables based on repo_path or db_path, and returns repo_id."""
    from archaeologist.storage.paths import get_default_qdrant_path
    from archaeologist.storage.context import (
        current_db_url_var,
        current_bm25_path_var,
        current_qdrant_path_var,
        current_repo_path_var
    )
    repo_id = None
    if repo_path:
        abs_repo = os.path.abspath(repo_path)
        os.environ["ARCHAEOLOGIST_REPO"] = abs_repo
        os.environ["DATABASE_URL"] = get_default_db_url(abs_repo)
        os.environ["BM25_INDEX_PATH"] = get_default_bm25_path(abs_repo)
        os.environ["QDRANT_STORAGE_PATH"] = get_default_qdrant_path(abs_repo)
        current_db_url_var.set(os.environ["DATABASE_URL"])
        current_bm25_path_var.set(os.environ["BM25_INDEX_PATH"])
        current_qdrant_path_var.set(os.environ["QDRANT_STORAGE_PATH"])
        current_repo_path_var.set(abs_repo)
        repo_id = get_stored_repo_id(abs_repo) or resolve_repo_id(abs_repo)
    elif db_path:
        abs_db = os.path.abspath(db_path)
        repo_dir = os.path.dirname(abs_db)
        os.environ["ARCHAEOLOGIST_REPO"] = repo_dir
        os.environ["DATABASE_URL"] = f"sqlite:///{abs_db}"
        os.environ["BM25_INDEX_PATH"] = get_default_bm25_path(repo_dir)
        os.environ["QDRANT_STORAGE_PATH"] = get_default_qdrant_path(repo_dir)
        current_db_url_var.set(os.environ["DATABASE_URL"])
        current_bm25_path_var.set(os.environ["BM25_INDEX_PATH"])
        current_qdrant_path_var.set(os.environ["QDRANT_STORAGE_PATH"])
        current_repo_path_var.set(repo_dir)
        repo_id = get_stored_repo_id(repo_dir)
    return repo_id

import sys

# Configure UTF-8 streams for cross-platform terminal output (preventing Windows cp1252 errors)
if sys.platform == "win32":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load local .env then sync global user configuration
load_dotenv()
sync_env_from_config()

app = typer.Typer(
    name="archaeologist",
    help="Codebase Archaeologist - Autonomous Forensic Code Intelligence CLI",
    add_completion=False
)

def _get_console():
    from rich.console import Console
    return Console()

@app.command("setup")
@app.command("init")
def setup():
    """Interactive wizard to configure API keys (Gemini & GitHub) and preferences."""
    from rich.prompt import Prompt
    from rich.panel import Panel
    console = _get_console()
    
    console.print(Panel.fit(
        "[bold cyan]🏺 Welcome to Codebase Archaeologist Setup Wizard[/bold cyan]\n"
        "Configure your keys once to use Codebase Archaeologist across any local repository.\n\n"
        "[bold]1. Google Gemini API Key[/bold] (Powers AI diff summaries, semantic embeddings, and 'arch ask')\n"
        "   Get a free key at: [cyan]https://aistudio.google.com/app/apikey[/cyan]\n\n"
        "[bold]2. GitHub Personal Access Token[/bold] (Optional: enriches history with PR reviews & Issue comments)\n"
        "   Get a token at: [cyan]https://github.com/settings/tokens[/cyan]",
        border_style="cyan"
    ))
    
    cfg = load_user_config()
    current_gemini = cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    current_github = cfg.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    
    prompt_gemini = f"Google Gemini API Key [{current_gemini[:4]}...{current_gemini[-4:]}]" if len(current_gemini) >= 8 else "Google Gemini API Key (press Enter to skip)"
    new_gemini = Prompt.ask(prompt_gemini, default=current_gemini, password=True, show_default=False)
    if new_gemini:
        cfg["GEMINI_API_KEY"] = new_gemini
        os.environ["GEMINI_API_KEY"] = new_gemini
        
    prompt_github = f"GitHub Personal Access Token (Optional) [{current_github[:4]}...]" if len(current_github) >= 4 else "GitHub Personal Access Token (Optional, press Enter to skip)"
    new_github = Prompt.ask(prompt_github, default=current_github, password=True, show_default=False)
    if new_github:
        cfg["GITHUB_TOKEN"] = new_github
        os.environ["GITHUB_TOKEN"] = new_github
        
    save_user_config(cfg)
    console.print("\n[bold green]✓ Configuration saved to ~/.archaeologist/config.json![/bold green]\n")

@app.command()
def ingest(
    repo_path: Optional[str] = typer.Argument(None, help="Path to local repository directory (defaults to current dir)"),
    repo_url: Optional[str] = typer.Option(None, help="GitHub repository URL (auto-detected from git remote if omitted)"),
    window: Optional[str] = typer.Option(
        None, "--window", "-w",
        help="Ingestion time window: '6m' (6 months), '1y' (1 year), '2y' (2 years), or 'full' (entire repo history)"
    ),
    since: Optional[str] = typer.Option(None, help="Specific start date for commits (YYYY-MM-DD)"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode alias for '--window 6m'"),
    gemini_key: Optional[str] = typer.Option(None, "--gemini-key", help="Google Gemini API key for semantic vectors and reasoning"),
    github_token: Optional[str] = typer.Option(None, "--github-token", help="GitHub Personal Access Token for remote PR/issue enrichment"),
    reembed: bool = typer.Option(False, "--reembed", help="Re-embed all chunks and recreate vector collection")
):
    """Ingests commits, PRs, issues, AST symbols, and reverts into local repository vector & metadata store."""
    from archaeologist.ingestion.pipeline import IngestionPipeline
    from rich.console import Console
    console = Console()
    
    target_path = repo_path or str(find_repo_root())
    if not os.path.exists(target_path):
        console.print(f"[bold red]Error:[/bold red] Repository path '{target_path}' does not exist.")
        raise typer.Exit(1)
        
    # Auto-detect GitHub remote URL if not provided
    detected_url = repo_url or detect_github_remote(target_path)
    
    # Process explicit CLI key arguments
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
        
    # Interactive key configuration check if running in a terminal
    if sys.stdin.isatty():
        from rich.prompt import Prompt
        cfg = load_user_config()
        
        # Check Gemini API Key
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            console.print("\n[bold yellow]Notice:[/bold yellow] No Google Gemini API key detected.")
            console.print("A Gemini key powers semantic embeddings, diff summaries, and 'arch ask' causal reasoning.")
            console.print("Get a free key at: [cyan]https://aistudio.google.com/app/apikey[/cyan]")
            entered_gemini = Prompt.ask("Enter Gemini API key (or press Enter for offline/keyword mode)", default="", password=True, show_default=False)
            if entered_gemini:
                os.environ["GEMINI_API_KEY"] = entered_gemini
                cfg["GEMINI_API_KEY"] = entered_gemini
                save_user_config(cfg)
                console.print("[green]✓ Saved Gemini API key to ~/.archaeologist/config.json[/green]\n")
            else:
                console.print("[dim]Continuing in offline mode with BM25 keyword search.[/dim]\n")
                
        # Check GitHub Token if linked to a remote GitHub repository
        if detected_url and not os.getenv("GITHUB_TOKEN"):
            console.print("[bold cyan]Tip:[/bold cyan] Linked to remote GitHub repository without a GITHUB_TOKEN.")
            console.print("A token enriches the causal graph with PR reviews and Issue comments. (Local git commits work 100% offline).")
            console.print("Get a token at: [cyan]https://github.com/settings/tokens[/cyan]")
            entered_gh = Prompt.ask("(Optional) Enter GitHub Personal Access Token (or press Enter to skip)", default="", password=True, show_default=False)
            if entered_gh:
                os.environ["GITHUB_TOKEN"] = entered_gh
                cfg["GITHUB_TOKEN"] = entered_gh
                save_user_config(cfg)
                console.print("[green]✓ Saved GitHub Token to ~/.archaeologist/config.json[/green]\n")
            else:
                console.print("[dim]Continuing with local git history only.[/dim]\n")
    
    from archaeologist.ingestion.git_parser import count_commits
    
    # Resolve ingestion window / since cutoff
    selected_window = window
    if quick:
        selected_window = "6m"
    elif not selected_window and not since:
        # If in an interactive terminal, offer user selection with live commit counts
        if sys.stdin.isatty():
            from rich.prompt import Prompt
            from rich.panel import Panel
            c_6m = count_commits(target_path, since=calculate_window_since("6m"))
            c_1y = count_commits(target_path, since=calculate_window_since("1y"))
            c_2y = count_commits(target_path, since=calculate_window_since("2y"))
            c_full = count_commits(target_path, since=None)
            
            console.print("\n[bold cyan]Select Ingestion Window:[/bold cyan]")
            console.print(f"  [white]1)[/white] [bold green]6m[/bold green]   - Last 6 months ([cyan]{c_6m:,} commits[/cyan]) [dim](fast, recommended for recent changes)[/dim]")
            console.print(f"  [white]2)[/white] [bold green]1y[/bold green]   - Last 1 year   ([cyan]{c_1y:,} commits[/cyan])")
            console.print(f"  [white]3)[/white] [bold green]2y[/bold green]   - Last 2 years  ([cyan]{c_2y:,} commits[/cyan])")
            console.print(f"  [white]4)[/white] [bold green]full[/bold green] - Full history ([cyan]{c_full:,} commits[/cyan])\n")
            selected_window = Prompt.ask("Choose window", choices=["6m", "1y", "2y", "full"], default="full")
        else:
            selected_window = "full"
            
    effective_since = since
    if selected_window and not effective_since:
        try:
            effective_since = calculate_window_since(selected_window)
        except ValueError as err:
            console.print(f"[bold red]Error:[/bold red] {err}")
            raise typer.Exit(1)
            
    total_repo_commits = count_commits(target_path, since=effective_since)
    console.print(f"🏺 Starting ingestion for repository: [bold cyan]{target_path}[/bold cyan]")
    if selected_window and selected_window.lower() in ("full", "all"):
        console.print(f"⏳ Ingestion Window: [bold yellow]Full Repository History[/bold yellow] ([cyan]{total_repo_commits:,} commits[/cyan])")
    elif effective_since:
        console.print(f"⏳ Ingestion Window: [bold yellow]{selected_window or 'custom'}[/bold yellow] ([cyan]{total_repo_commits:,} commits[/cyan] on or after {effective_since})")
    if detected_url:
        console.print(f"🔗 Linked GitHub Remote: [green]{detected_url}[/green]")
        
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )
    
    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.fields[phase_tag]}[/bold cyan]"),
        BarColumn(bar_width=32, style="dim white", complete_style="bold green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[white]{task.fields[status_msg]}[/white]"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    ) as progress:
        main_task = progress.add_task(
            "Ingestion",
            total=100.0,
            completed=0.0,
            phase_tag="🏺 [  0%]",
            status_msg="Starting ingestion engine..."
        )

        def on_progress(phase: str, current: int, total: int, message: str, percent: float):
            pct_str = f"🏺 [{int(percent):>3d}%]"
            progress.update(
                main_task,
                completed=percent,
                phase_tag=pct_str,
                status_msg=message
            )

        pipeline = IngestionPipeline(
            repo_path=target_path,
            repo_url=detected_url,
            since_date=effective_since,
            progress_callback=on_progress,
            github_token=os.getenv("GITHUB_TOKEN"),
            gemini_key=os.getenv("GEMINI_API_KEY"),
            reembed=reembed
        )
        try:
            pipeline.run()
        except Exception as e:
            console.print(f"\n[bold red]Error running ingestion pipeline:[/bold red] {e}")
            raise typer.Exit(1)
            
    console.print("\n[bold green]✓ Ingestion completed successfully![/bold green]")
    console.print("You can now run: [bold cyan]arch ask \"Why did ... change?\"[/bold cyan]")

@app.command()
def status(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON")
):
    """Shows repository index statistics and health status."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    target_repo = repo or "."
    resolved_root = str(find_repo_root(target_repo))
    db_file = db_path or get_default_db_path(target_repo)
    console = _get_console()

    if not os.path.exists(db_file):
        if as_json:
            import json
            print(json.dumps({"status": "unindexed", "repo_path": resolved_root, "db_path": db_file}))
        else:
            console.print(f"[bold yellow]Notice:[/bold yellow] No index found for '{resolved_root}' (at {db_file}).")
            console.print("Run [bold cyan]'archaeologist ingest'[/bold cyan] first to index this repository.")
        return

    from archaeologist.storage.db import get_session_context
    from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex, RepoMeta
    from sqlmodel import select, func

    with get_session_context() as session:
        c_count = session.exec(select(func.count(Commit.sha))).one() or 0
        pr_count = session.exec(select(func.count(PullRequest.number))).one() or 0
        issue_count = session.exec(select(func.count(Issue.number))).one() or 0
        chunk_count = session.exec(select(func.count(Chunk.id))).one() or 0
        embedded_count = session.exec(select(func.count(Chunk.id)).where(Chunk.embedded == True)).one() or 0
        symbol_count = session.exec(select(func.count(SymbolIndex.symbol_id))).one() or 0
        meta_dict = {}
        try:
            meta_stmt = select(RepoMeta)
            if repo_id:
                meta_stmt = meta_stmt.where(RepoMeta.repo_id == repo_id)
            meta_row = session.exec(meta_stmt).first() or session.exec(select(RepoMeta)).first()
            if meta_row:
                if meta_row.repo_id:
                    meta_dict["repo_id"] = meta_row.repo_id
                if meta_row.embedder_provider:
                    meta_dict["embedder_provider"] = meta_row.embedder_provider
                if meta_row.embedder_dimension:
                    meta_dict["embedder_dimension"] = str(meta_row.embedder_dimension)
                if meta_row.last_ingested_at:
                    meta_dict["last_ingested_at"] = str(meta_row.last_ingested_at)
                elif meta_row.created_at:
                    meta_dict["last_ingested_at"] = str(meta_row.created_at)
            for m in session.exec(select(RepoMeta)).all():
                if m.key and m.value:
                    meta_dict[m.key] = m.value
        except Exception:
            pass

        if "embedder_provider" not in meta_dict or meta_dict["embedder_provider"] == "none":
            from archaeologist.storage.paths import get_repo_data_dir
            meta_json_path = os.path.join(get_repo_data_dir(resolved_root), "meta.json")
            if os.path.exists(meta_json_path):
                try:
                    with open(meta_json_path, "r", encoding="utf-8") as f:
                        file_meta = json.load(f)
                    for k, v in file_meta.items():
                        if k not in meta_dict and v is not None:
                            meta_dict[k] = str(v)
                except Exception:
                    pass

    data = {
        "status": "ready",
        "repo_id": repo_id or meta_dict.get("repo_id", "unknown"),
        "repo_path": resolved_root,
        "db_path": db_file,
        "total_commits": c_count,
        "total_pull_requests": pr_count,
        "total_issues": issue_count,
        "total_chunks": chunk_count,
        "embedded_chunks": embedded_count,
        "embedded_percentage": round((embedded_count / chunk_count * 100), 1) if chunk_count else 0.0,
        "total_ast_symbols": symbol_count,
        "embedding_provider": meta_dict.get("embedder_provider", "none"),
        "embedding_dimension": meta_dict.get("embedder_dimension", "none"),
        "last_ingested_at": meta_dict.get("last_ingested_at", "unknown")
    }

    if as_json:
        import json
        print(json.dumps(data, indent=2))
        return

    from rich.table import Table
    from rich import box

    table = Table(
        title="🏺 Repository Index Health & Forensic Statistics",
        border_style="bright_blue",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Repository ID", str(data["repo_id"]))
    table.add_row("Repository Root", str(data["repo_path"]))
    table.add_row("Database Path", str(data["db_path"]))
    table.add_row("Commits Indexed", f"{data['total_commits']:,}")
    table.add_row("Pull Requests", f"{data['total_pull_requests']:,}")
    table.add_row("GitHub Issues", f"{data['total_issues']:,}")
    table.add_row("AST Symbols", f"{data['total_ast_symbols']:,}")
    table.add_row("History Chunks", f"{data['total_chunks']:,} ({data['embedded_chunks']:,} vector-indexed, {data['embedded_percentage']}%)")
    table.add_row("Vector Backend", f"{data['embedding_provider']} (dim: {data['embedding_dimension']})")
    table.add_row("Last Ingested", str(data["last_ingested_at"]))

    console.print(table)

@app.command()
def ask(
    question: str = typer.Argument(..., help="The causal query about the codebase"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path")
):
    """Asks a causal query using the full LangGraph agent loop powered by Gemini."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.mcp_server.tools import ask_tool
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich import box
    console = Console()
    
    console.print(f"\n[bold cyan]🔍 Forensic Investigation:[/bold cyan] [bold white]{question}[/bold white]")
    with console.status("[bold green]Running LangGraph multi-hop forensic reasoning agent...[/bold green]"):
        try:
            res = ask_tool(question, repo_id=repo_id)
        except Exception as e:
            console.print(f"\n[bold red]Error during forensic investigation:[/bold red] {e}")
            raise typer.Exit(1)
        
    console.print("")
    console.print(Panel(
        Markdown(res),
        title="[bold green]🏺 Forensic Analysis & Causal Rationale[/bold green]",
        subtitle="[dim]Grounded in git commits, diffs, AST symbols & PR discussions[/dim]",
        border_style="bright_green",
        box=box.ROUNDED,
        padding=(1, 2)
    ))

@app.command()
def hotspots(
    top_n: int = typer.Option(15, help="Top N hotspot files to list"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of tables")
):
    """Lists repository hotspot files ranked by commit frequency."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.storage.analytics import repo_hotspots_tool
    from rich.table import Table
    from rich import box
    console = _get_console()
    
    res = repo_hotspots_tool(top_n=top_n, repo_id=repo_id)
    hotspots_list = res if isinstance(res, list) else res.get("hotspots", [])
    if as_json:
        import json
        print(json.dumps(hotspots_list, indent=2, default=str))
        return
    if not hotspots_list:
        console.print("[bold yellow]Notice:[/bold yellow] No commits found for this repository. Run [bold cyan]'archaeologist ingest'[/bold cyan] first.")
        return
    
    max_c = max([item.get("commit_count", 0) for item in hotspots_list] or [1])
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
    
    for idx, item in enumerate(hotspots_list, 1):
        c_count = item.get("commit_count", 0)
        bar_len = max(1, int((c_count / max_c) * 16)) if max_c > 0 else 1
        bar_visual = "█" * bar_len
        table.add_row(f"#{idx}", item.get("file_path", ""), f"{c_count:,}", f"[magenta]{bar_visual}[/magenta]")
        
    console.print(table)

@app.command()
def ownership(
    file_path: Optional[str] = typer.Option(None, help="Optional file path to inspect"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of tables")
):
    """Analyzes author contribution distribution and bus factor risk."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.storage.analytics import repo_ownership_tool
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = _get_console()
    
    res = repo_ownership_tool(file_path=file_path, repo_id=repo_id)
    if as_json:
        import json
        print(json.dumps(res, indent=2, default=str))
        return

    title = f"👥 Code Ownership: {file_path}" if file_path else "👥 Repository-Wide Author Contribution Distribution"
    table = Table(
        title=title,
        border_style="cyan",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_cyan"
    )
    table.add_column("Author", style="bold white")
    table.add_column("Commits", justify="right", style="cyan")
    table.add_column("Share %", justify="right", style="bold green")
    table.add_column("Contribution", justify="left", style="green")
    
    distribution = res.get("author_distribution", {})
    if not distribution:
        console.print("[bold yellow]Notice:[/bold yellow] No commits found for this repository. Run [bold cyan]'archaeologist ingest'[/bold cyan] first.")
        return
    sorted_authors = sorted(distribution.items(), key=lambda x: x[1].get("commit_count", 0), reverse=True)
    for author, stats in sorted_authors[:15]:
        pct = stats.get('percentage', 0.0)
        bar_len = max(1, int(pct / 100 * 16))
        bar_visual = "█" * bar_len
        table.add_row(
            author,
            str(stats.get("commit_count", 0)),
            f"{pct:.1f}%",
            f"[green]{bar_visual}[/green]"
        )
        
    console.print(table)
    if res.get("bus_factor_risk") == "HIGH (Single Author Dominance)":
        console.print(Panel(
            "[bold red]⚠️  High Bus Factor Risk Detected:[/bold red] A single author accounts for >60% of all changes.\n"
            "[dim]Recommendation: Distribute code reviews and documentation to avoid single-point knowledge loss.[/dim]",
            border_style="bright_red",
            box=box.ROUNDED
        ))

@app.command()
def coupling(
    min_co_commits: int = typer.Option(2, help="Minimum co-commits required"),
    top_n: int = typer.Option(15, help="Top N coupled file pairs to list"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of tables")
):
    """Identifies pairs of files that frequently change together (temporal coupling)."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.storage.analytics import change_coupling_tool
    from rich.table import Table
    from rich import box
    console = _get_console()
    
    res = change_coupling_tool(min_co_commits=min_co_commits, top_n=top_n, repo_id=repo_id)
    pairs = res if isinstance(res, list) else res.get("coupled_pairs", [])
    if as_json:
        import json
        print(json.dumps(pairs, indent=2, default=str))
        return
    if not pairs:
        console.print("[bold yellow]Notice:[/bold yellow] No coupled file changes found. Run [bold cyan]'archaeologist ingest'[/bold cyan] or lower '--min-co-commits'.")
        return
    
    max_co = max([item.get("co_commit_count", 0) for item in pairs] or [1])
    table = Table(
        title="🔗 Temporal Change Coupling (Files Changed Together)",
        border_style="magenta",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_magenta"
    )
    table.add_column("Rank", justify="center", style="bold cyan", no_wrap=True)
    table.add_column("File A", style="bright_white")
    table.add_column("File B", style="bright_white")
    table.add_column("Co-Commits", justify="right", style="bold green")
    table.add_column("Coupling Strength", justify="left", style="green")
    
    for idx, item in enumerate(pairs, 1):
        co_c = item.get("co_commit_count", 0)
        bar_len = max(1, int((co_c / max_co) * 12)) if max_co > 0 else 1
        table.add_row(
            f"#{idx}",
            item.get("file_a", ""),
            item.get("file_b", ""),
            str(co_c),
            f"[green]{'█' * bar_len}[/green]"
        )
        
    console.print(table)

@app.command()
def symbols(
    top_n: int = typer.Option(20, help="Top N AST symbols to list"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of tables")
):
    """Lists extracted AST Code Symbols (classes, functions, methods) ranked by commit count."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.storage.analytics import repo_symbols_tool
    from rich.table import Table
    from rich import box
    console = _get_console()
    
    res = repo_symbols_tool(top_n=top_n, repo_id=repo_id)
    syms = res if isinstance(res, list) else res.get("symbols", [])
    if as_json:
        import json
        print(json.dumps(syms, indent=2, default=str))
        return
    if not syms:
        console.print("[bold yellow]Notice:[/bold yellow] No AST symbols found for this repository. Run [bold cyan]'archaeologist ingest'[/bold cyan] first.")
        return
    
    table = Table(
        title="🧩 AST Code Symbols by Modification Frequency",
        border_style="blue",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_blue"
    )
    table.add_column("Symbol Name", style="bold cyan")
    table.add_column("Kind", style="yellow")
    table.add_column("File Path", style="dim white")
    table.add_column("Modifications", justify="right", style="bold green")
    
    for s in syms:
        table.add_row(
            s.get("symbol_name", ""),
            s.get("kind", ""),
            s.get("file_path", ""),
            str(s.get("commit_count", 0))
        )
        
    console.print(table)

@app.command()
def symbol_history(
    symbol_query: str = typer.Argument(..., help="Class or function name to search"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    db_path: Optional[str] = typer.Option(None, help="Optional SQLite database path"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of tables")
):
    """Retrieves all commits that modified a specific AST Code Symbol."""
    repo_id = _configure_repo_env(repo_path=repo, db_path=db_path)
    from archaeologist.storage.analytics import symbol_history_tool
    from rich.table import Table
    from rich import box
    console = _get_console()
    
    commits = symbol_history_tool(symbol_query=symbol_query, repo_id=repo_id)
    if as_json:
        import json
        print(json.dumps(commits, indent=2, default=str))
        return
    if not commits:
        console.print(f"[bold yellow]Notice:[/bold yellow] No commits found modifying AST symbol [bold cyan]'{symbol_query}'[/bold cyan].")
        return
    
    table = Table(
        title=f"📜 Historical Modifications for Symbol: [bold cyan]{symbol_query}[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        header_style="bold bright_white on dark_cyan"
    )
    table.add_column("SHA", style="bold yellow", no_wrap=True)
    table.add_column("Date", style="dim", no_wrap=True)
    table.add_column("Author", style="cyan")
    table.add_column("Commit Message", style="white")
    
    for c in commits:
        table.add_row(
            c.get("sha", "")[:8],
            str(c.get("date", ""))[:10],
            c.get("author", ""),
            (c.get("message", "") or "").splitlines()[0][:60]
        )
        
    console.print(table)

@app.command()
def why(
    target: str = typer.Argument(..., help="File and line number to explain (e.g. 'src/auth.py:42' or 'auth.py:10-20')"),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path (defaults to current repo)"),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON instead of panel")
):
    """Explains why a specific line or range of code exists using git blame and causal context."""
    from archaeologist.mcp_server.tools import blame_explain_tool
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich import box
    console = _get_console()

    if ":" not in target:
        console.print("[bold red]Error:[/bold red] Target must be in 'file:line' format (e.g. 'src/auth.py:42' or 'src/auth.py:10-25').")
        raise typer.Exit(1)

    fpath, lrange = target.rsplit(":", 1)
    if "-" in lrange:
        try:
            s_str, e_str = lrange.split("-", 1)
            l_start, l_end = int(s_str), int(e_str)
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid line range '{lrange}'.")
            raise typer.Exit(1)
    else:
        try:
            l_start = l_end = int(lrange)
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid line number '{lrange}'.")
            raise typer.Exit(1)

    repo_dir = repo or "."
    res = blame_explain_tool(repo_path=repo_dir, file_path=fpath, line_start=l_start, line_end=l_end)
    if as_json:
        import json
        print(json.dumps(res, indent=2, default=str))
        return

    explanation = res.get("explanation", "No history found.")
    commits = res.get("commits", [])

    console.print(f"\n[bold cyan]🔍 Causal Investigation:[/bold cyan] [bold white]{target}[/bold white]")
    if commits:
        console.print(f"[dim]Identified {len(commits)} relevant commit(s) touching lines {l_start}-{l_end}:[/dim]")
        for c in commits[:5]:
            console.print(f"  • [bold yellow]{c.get('sha', '')[:7]}[/bold yellow] {c.get('message', '')} [dim]({c.get('author', '')})[/dim]")

    console.print("")
    console.print(Panel(
        Markdown(explanation),
        title="[bold green]🏺 Forensic Blame Explanation[/bold green]",
        subtitle=f"[dim]{target}[/dim]",
        border_style="bright_green",
        box=box.ROUNDED,
        padding=(1, 2)
    ))

@app.command()
def ui(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path to inspect (defaults to current repo)"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind"),
    port: int = typer.Option(8000, help="Port to bind the web server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    open_browser: bool = typer.Option(True, help="Automatically open browser on launch")
):
    """Starts the Codebase Archaeologist Interactive Causal Graph Web UI."""
    import uvicorn
    import webbrowser
    import threading
    if repo:
        _configure_repo_env(repo_path=repo)
    console = _get_console()
    
    url = f"http://{host}:{port}"
    console.print(f"🏺 Starting Codebase Archaeologist Causal Graph UI at [bold cyan]{url}[/bold cyan]...")
    
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        
    uvicorn.run("archaeologist.web.server:app", host=host, port=port, reload=reload)

@app.command()
def serve(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path to inspect (defaults to current repo)"),
    host: str = typer.Option("127.0.0.1", help="Host address to bind"),
    port: int = typer.Option(8000, help="Port to bind the web server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development")
):
    """Starts the Codebase Archaeologist Interactive Web UI & API Server (headless)."""
    import uvicorn
    if repo:
        _configure_repo_env(repo_path=repo)
    console = _get_console()
    console.print(f"🏺 Starting Codebase Archaeologist Server at http://{host}:{port}...")
    uvicorn.run("archaeologist.web.server:app", host=host, port=port, reload=reload)

@app.command()
def mcp(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path to serve (defaults to current repo)")
):
    """Starts the Codebase Archaeologist Native MCP Server on stdio transport."""
    target_repo = repo or "."
    _configure_repo_env(repo_path=target_repo)
    from archaeologist.storage.paths import get_default_db_path
    db_file = get_default_db_path(target_repo)
    if not os.path.exists(db_file):
        import sys
        print(f"Error: No archaeologist index found at {db_file}. Please run 'archaeologist ingest {target_repo}' first.", file=sys.stderr)
        raise typer.Exit(1)
    from archaeologist.mcp_server.server import mcp as mcp_instance
    import sys
    print(f"Starting stdio MCP server for {target_repo}...", file=sys.stderr)
    mcp_instance.run()

@app.command()
def start_server(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository path to serve (defaults to current repo)")
):
    """Starts the Codebase Archaeologist MCP Server (alias for mcp)."""
    mcp(repo=repo)

@app.command()
def run_eval(
    dataset: Optional[str] = typer.Option(None, help="Path to QA pairs dataset JSONL file")
):
    """Runs the evaluation harness to measure Grounded Accuracy and Citation Precision."""
    from eval.run_eval import run_evaluation
    console = _get_console()
    console.print(f"Starting evaluation suite (dataset: [cyan]{dataset or 'default'}[/cyan])...")
    run_evaluation(dataset)

if __name__ == "__main__":
    app()
