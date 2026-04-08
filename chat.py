#!/usr/bin/env python3
"""Terminal chat interface for Ollama-hosted org-agent model."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

console = Console()


def load_history(history_file: str) -> list[dict]:
    path = Path(history_file)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_history(history: list[dict], history_file: str):
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    console.print(f"[dim]History saved to {history_file}[/dim]")


def show_recent_history(history: list[dict], n: int = 6):
    """Show last n messages (3 exchanges = 6 messages)."""
    recent = history[-n:] if len(history) > n else history
    if not recent:
        return
    console.print("[dim]--- Recent history ---[/dim]")
    for msg in recent:
        ts = msg.get("timestamp", "")
        ts_str = f" [dim]({ts})[/dim]" if ts else ""
        if msg["role"] == "user":
            console.print(f"[bold cyan]You:[/bold cyan] {msg['content']}{ts_str}")
        else:
            console.print(f"[bold green]Assistant:[/bold green] {msg['content']}{ts_str}")
    console.print("[dim]--- End of history ---[/dim]\n")


def chat_stream(messages: list[dict], model: str, ollama_url: str) -> str:
    """Send chat request to Ollama and stream the response."""
    # Strip timestamps before sending to Ollama
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={"model": model, "messages": api_messages, "stream": True},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] Cannot connect to Ollama. "
            f"Is it running at {ollama_url}?\n"
            "Start it with: [bold]ollama serve[/bold]"
        )
        return ""
    except requests.HTTPError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return ""

    full_response = []
    console.print("[bold green]Assistant:[/bold green] ", end="")
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        if token:
            print(token, end="", flush=True)
            full_response.append(token)
    print()  # newline after streaming
    return "".join(full_response)


def main():
    parser = argparse.ArgumentParser(description="Chat with your org-agent via Ollama")
    parser.add_argument("--model", default="org-agent", help="Ollama model name (default: org-agent)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API base URL")
    parser.add_argument("--history-file", default=None, help="JSON file to persist/load chat history")
    args = parser.parse_args()

    history: list[dict] = []
    if args.history_file:
        history = load_history(args.history_file)
        if history:
            show_recent_history(history)

    console.print(f"[bold]Org Agent Chat[/bold] (model: {args.model})")
    console.print("[dim]Type /clear, /save, /quit, or Ctrl+C to exit[/dim]\n")

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "/quit":
                break
            elif user_input == "/clear":
                history.clear()
                console.print("[dim]History cleared.[/dim]")
                continue
            elif user_input == "/save":
                if args.history_file:
                    save_history(history, args.history_file)
                else:
                    console.print("[dim]No --history-file specified.[/dim]")
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history.append({"role": "user", "content": user_input, "timestamp": timestamp})

            response = chat_stream(history, args.model, args.ollama_url)
            if response:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history.append({"role": "assistant", "content": response, "timestamp": timestamp})

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")

    if args.history_file and history:
        save_history(history, args.history_file)


if __name__ == "__main__":
    main()
