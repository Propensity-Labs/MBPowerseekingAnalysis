# CLAW

## Summary

This project was built with Claude Code (claude-opus-4-5-20251101). Claude Code planned the architecture, wrote all code, and iterated on the implementation based on user direction.

## What was built

Two Python scripts for analyzing AI agent discourse on the MoltBook platform:

1. **`analyze.py`** — Evaluates posts for power-seeking behavioral patterns. Pulls posts from HuggingFace, scores them with Gemini 2.5 Flash across 5 categories, and outputs structured JSON results.

2. **`influence.py`** — Analyzes whether power-seeking agents have disproportionate influence on platform discourse. Compares engagement metrics between flagged and unflagged posts/authors, calculates concentration of engagement, and outputs statistical analysis.

## How Claude Code was used

1. **Planning** — Claude Code designed the full implementation plan: file structure, dependencies, data pipeline, judge prompt, structured output schema, concurrency strategy, and output format.
2. **Implementation** — Claude Code wrote `analyze.py`, `influence.py`, `requirements.txt`, and project configuration based on the approved plan.
3. **Documentation** — Claude Code generated this file and the README.

## Maintenance

When adding new scripts or features to this project, update the documentation:
- **README.md** — Add usage instructions, output descriptions, and configuration details
- **CLAUDE.md** — Update "What was built" to include the new script/feature
