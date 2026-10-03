# Mirumi — Local Qwen Companion

Mirumi is a local/offline character conversation program. Python coordinates
character files, bounded conversation context, structured external memory,
and the existing Qwen GGUF model through `llama.cpp`. It does **not** train or
modify Qwen's model weights.

## What the original code did

The original `mirumi.py` launched `llama-cli` with a 1,024-token context,
loaded `core.txt` and `character_rules.txt`, and kept only the current process's
last six message objects in RAM. It appended user messages to session JSONL
files, but the assistant-save call was part of a comment, so assistant replies
were not persisted. On every user turn it started the model a second time to
analyze self-development, then wrote model-selected traits directly to numbered
permanent files. The separate identity, personality, behavior, speech, and
stable-fact files were not loaded by that controller. Old session files were
not restored into a new process. Several copies of the nature preference were
already present in legacy development files.

This implementation keeps those source files and old session records intact.
It does not migrate or rewrite them on startup.

## Architecture

- `mirumi/model.py` defines a small inference interface and a `llama-cli`
  implementation. The same Qwen GGUF and llama.cpp toolchain remain in use.
- `mirumi/character.py` loads identity, personality, behavior, speech style,
  stable facts, and character rules separately. The short `core.txt` remains a
  compatibility fallback; source character files are not rewritten.
- `mirumi/conversation.py` appends both user and assistant turns to JSONL,
  restores only bounded recent context, reads legacy `mirumi` assistant roles,
  and skips malformed records without discarding valid lines.
- `mirumi/memory.py` stores typed records with source, confidence, status,
  evidence, and append-only review decisions. Retrieval is local lexical
  relevance matching; there are no embeddings, cloud services, or added
  packages.
- `mirumi/context.py` assembles a bounded prompt from character configuration,
  a few approved relevant memories, a context-only summary, and recent turns.
  Older context is removed from the prompt only; conversation files remain.
- `mirumi/reasoning.py` identifies explicit task lists. The normal response is
  one generation; if an explicit checklist is clearly incomplete, a targeted
  revision asks only for the omitted requirements rather than repeating the
  same question.
- `mirumi/backup.py` creates a local copy of memory and legacy development
  files. No migration currently runs automatically, so there is no startup
  rewrite that needs a pre-migration backup.

The small local model can still make mistakes. Retrieval reduces irrelevant
context; it does not guarantee that a model-generated answer is true.

## Memory and controlled development

Memory is external data supplied in the prompt, not learning in Qwen's weights.
Normal assistant output is never automatically promoted to a fact.

- `/remember [category:] text` records only information explicitly selected by
  the user as an approved memory. Categories include `fact`, `preference`,
  `relationship`, `event`, `episodic`, `behavior`, and `temporary`.
- `/memories [query]` shows approved memories or retrieves relevant ones.
- `/reflect` asks the local model to look for a repeated Mirumi behavior. It
  requires several recorded assistant turns, a verbatim quote from the actual
  transcript, and creates a low-confidence **pending** proposal only.
- `/pending`, `/approve ID`, and `/reject ID` make review explicit. Approval is
  an append-only decision, not a rewrite of `identity.txt` or other character
  files. Approved learned behavior stays separately labeled as developed.
- Existing `self_developed.txt`, `development_history.txt`, and
  `memory/self_development/*.txt` entries are read as low-confidence legacy
  candidates. Duplicate wording is grouped for review; none is injected as a
  confirmed memory unless approved.
- `/forget ID` appends a decision that excludes the record from retrieval; it
  does not erase its record or its source.
- `/summarize` stores a compact local-model summary as **context only**. The
  source conversation remains in its original JSONL file.
- `/backup` copies current memory and legacy development files under
  `memory/backups/`.

New runtime JSONL files, future conversation sessions, and local backups are
ignored by Git. The already tracked legacy conversation file remains in
history; it is not deleted or rewritten.

## Android / Termux setup

The repository contains no GGUF model weights and adds no Python dependencies.
Install Python and build the checked-in llama.cpp submodule once:

```sh
pkg update
pkg install python git cmake ninja clang make
cd ~/my-ai
git submodule update --init llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-cli -j2
mkdir -p ~/models
```

Place `Qwen3-1.7B-Q4_K_M.gguf` at `~/models/Qwen3-1.7B-Q4_K_M.gguf`, or set
`MIRUMI_MODEL` to its actual local path. Then configure and run:

```sh
export MIRUMI_HOME="$HOME/my-ai"
export MIRUMI_LLAMA_CLI="$HOME/my-ai/llama.cpp/build/bin/llama-cli"
python "$MIRUMI_HOME/mirumi.py"
```

`MIRUMI_HOME` defaults to the directory containing the checked-out project.
`MIRUMI_MODEL` and `MIRUMI_LLAMA_CLI` default to the paths above. Other
supported settings:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `MIRUMI_CONTEXT_SIZE` | `2048` | llama.cpp context tokens (allowed range 1,024–8,192) |
| `MIRUMI_MAX_TOKENS` | `180` | Maximum generated response tokens |
| `MIRUMI_THREADS` | up to `4` | CPU threads; lower this if the phone becomes hot |
| `MIRUMI_TIMEOUT_SECONDS` | `180` | Per-inference timeout |
| `MIRUMI_MAX_HISTORY_CHARS` | `1800` | In-process recent transcript budget |

For a 4 GB device, start at the defaults or reduce context/threads if other apps
are being killed. Larger contexts use more memory; the program uses a
conservative character-based prompt budget and does not assume a GPU. After
the model and llama.cpp are installed, conversation, memory, summaries, and
inference work offline.

## Commands

Type `/help` in a session for the full command list. The main commands are
`/remember`, `/memories`, `/pending`, `/approve ID`, `/reject ID`,
`/forget ID`, `/reflect`, `/summarize`, `/backup`, and `/exit`.

## Tests

Tests use only the Python standard library and do not load the GGUF:

```sh
python -m unittest discover -s tests -v
```

They cover memory validation and retrieval, legacy-data preservation,
malformed JSONL recovery, bounded prompt assembly, targeted checklist review,
local-model error handling, and persistence of assistant turns.