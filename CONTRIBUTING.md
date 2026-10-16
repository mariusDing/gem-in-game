# Contributing to hoop-tape-miner

Thanks for helping. Fixes, new league importers, better prompts, docs and bug reports are all welcome.

## Ground rule: no real game data

This project analyses real people, often amateurs and sometimes minors. **Never attach game videos, frames, screenshots of reports, rosters, player names or official stats to an issue, pull request or discussion.** Describe the problem in words, or reproduce it with synthetic data. Details: [docs/privacy.md](docs/privacy.md).

## Ways to help

- **Questions and ideas:** [Discussions](https://github.com/mariusDing/hoop-tape-miner/discussions).
- **Bugs:** open an issue with the bug form. Include the stage, the command and the traceback.
- **Features:** open an issue first so we can agree on the approach before you write code.
- **First contribution?** Pick a [good first issue](https://github.com/mariusDing/hoop-tape-miner/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

## Development setup

You need [uv](https://docs.astral.sh/uv/), Python 3.11 and `ffmpeg`. The full pipeline needs an NVIDIA GPU; the tests do not.

```bash
git clone https://github.com/<you>/hoop-tape-miner.git
cd hoop-tape-miner
uv sync
uv run ruff check .
uv run pytest -q
```

Without a CUDA GPU, install CPU torch instead of running `uv sync`, as CI does:

```bash
uv venv --python 3.11
uv pip install --no-sources --torch-backend cpu -e . pytest
uv run --no-sync pytest -q
```

## Project map

- `src/hoop_tape_miner/` the pipeline, one module per CLI stage; `cli.py` lists them.
- `docs/playbook.md` the procedure an AI agent follows, video to report. `docs/pipeline.md` has stage details.
- `prompts/` writing guides for the agent (film review, coach report, narration).
- `templates/` `game.json` and report skeletons.
- `tests/` fast tests with small synthetic fixtures. No GPU, no video.

## Pull requests

1. Fork, then branch from `master` (`fix/jersey-crop`, `feat/league-importer-x`).
2. Keep a PR to one change. Add or update tests for behaviour changes.
3. Run `uv run ruff check .` and `uv run pytest -q` before you push. CI runs both checks.
4. Update the docs and `CHANGELOG.md` (under *Unreleased*) when users would notice the change.
5. Check `git status`: nothing from `workspace/`, no videos, frames or audio.

Code style: follow the surrounding code. Small functions, plain names, short docstrings that say what a stage reads and writes.

## AI agents welcome

Using Claude Code, Codex or another agent to contribute is fine. Agents read [AGENTS.md](AGENTS.md). You are responsible for what the agent submits: review the diff and run the tests.

## Code of Conduct

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
