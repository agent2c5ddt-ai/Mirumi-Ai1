import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

BASE = Path.home() / "my-ai"
LLAMA = Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli"
MODEL = Path.home() / "models" / "Qwen3-1.7B-Q4_K_M.gguf"

CORE = BASE / "core.txt"
RULES = BASE / "character_rules.txt"

CONV_DIR = BASE / "memory" / "conversations"
DEV_DIR = BASE / "memory" / "self_development"

CONV_DIR.mkdir(parents=True, exist_ok=True)
DEV_DIR.mkdir(parents=True, exist_ok=True)

def read_file(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def load_developments():
    entries = []
    for path in sorted(DEV_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            entries.append(text)
    return "\n\n".join(entries)

def call_model(system_text, prompt, max_tokens=180):
    cmd = [
        str(LLAMA),
        "-m", str(MODEL),
        "-c", "1024",
        "-b", "128",
        "-ub", "64",
        "-t", "4",
        "-n", str(max_tokens),
        "--no-show-timings",
        "-sys", system_text,
        "-p", prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "llama-cli failed")

    return result.stdout.strip()

def save_message(conversation_file, role, text):
    record = {
        "time": now(),
        "role": role,
        "text": text,
    }

    with conversation_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def analyze_development(user_text, assistant_text, existing):
    analyzer = """You are Mirumi's development analyzer.

Your job is NOT to roleplay.
Your job is to identify whether this conversation contains a genuine
new development about Mirumi herself.

A development can be:
- a new preference
- a new dislike
- a new habit
- a new recurring tendency
- a new opinion
- a new self-discovered association
- a meaningful change in how Mirumi understands herself

Do NOT save:
- temporary wording
- random guesses
- facts merely suggested by the user
- things Mirumi said only because the prompt required them
- generic AI behavior
- one-off nonsense

Return exactly one of these:

KEEP=NO

or

KEEP=YES
TYPE=<preference|habit|opinion|association|other>
TEXT=<one concise statement about Mirumi's development>

Never output anything else.
"""

    prompt = f"""
Existing self-developed traits:

{existing if existing else "(none)"}

User:
{user_text}

Mirumi:
{assistant_text}

Did Mirumi genuinely develop something new about herself?
"""

    result = call_model(analyzer, prompt, max_tokens=80)

    if "KEEP=YES" not in result:
        return

    lines = {}
    for line in result.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            lines[key.strip()] = value.strip()

    text = lines.get("TEXT")
    if not text:
        return

    number = len(list(DEV_DIR.glob("*.txt"))) + 1
    path = DEV_DIR / f"{number:04d}.txt"

    content = f"""SELF-DEVELOPMENT {number:04d}

Date: {now()}
Type: {lines.get("TYPE", "other")}

Development:
{text}

Origin:
This development was identified from an actual conversation.

Status:
Preserved as a self-developed trait.
"""

    path.write_text(content, encoding="utf-8")

def main():
    conversation_file = (
        CONV_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    core = read_file(CORE)
    rules = read_file(RULES)

    print("Mirumi local session")
    print("Commands: /exit, /memory, /developments")
    print()

    history = []

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user:
            continue

        if user == "/exit":
            break

        if user == "/memory":
            print(load_developments() or "(no self-developed memories yet)")
            continue

        if user == "/developments":
            files = sorted(DEV_DIR.glob("*.txt"))
            if not files:
                print("(no developments yet)")
            else:
                for f in files:
                    print(f"\n--- {f.name} ---")
                    print(f.read_text(encoding="utf-8").strip())
            continue

        save_message(conversation_file, "user", user)
        history.append(("User", user))

        # Keep recent conversation small enough for the 1024-token context.
        recent = history[-6:]
        conversation = "\n".join(
            f"{role}: {text}" for role, text in recent
        )

        developments = load_developments()

        system = f"""{core}

{rules}

SELF-DEVELOPED MEMORIES
These are things Mirumi has developed through previous conversations.
Treat them as part of her continuing development.

{developments if developments else "(none yet)"}
"""

        prompt = f"""Continue this conversation naturally as Mirumi.

Recent conversation:
{conversation}

Respond only to the user's latest message.
"""

        try:
            answer = call_model(system, prompt, max_tokens=180)
        except Exception as e:
            print(f"[error] {e}")
            continue

        # The CLI may include extra formatting. Keep the text assave_message(conversation_file, "mirumi", answer)
        history.append(("Mirumi", answer))

        try:
            analyze_development(user, answer, developments)
        except Exception as e:
            print(f"[memory analyzer warning] {e}")

if __name__ == "__main__":
    main()
