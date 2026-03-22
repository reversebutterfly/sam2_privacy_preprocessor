# Project Context

This project implements a SAM2 privacy preprocessor.

## Goals
- Turn IDEA_SUMMARY.md into runnable experiments
- Reuse existing code whenever possible
- Do not rebuild the whole training framework
- Start with sanity checks before full experiments
- Save all experiment results to JSON or CSV

## Working Files
- Primary idea doc: IDEA_SUMMARY.md

## Local Environment
- OS: Windows
- Project root: E:\PycharmProjects\pythonProject\sam2_privacy_preprocessor

## Rules
- Read IDEA_SUMMARY.md before proposing code or experiments
- Prefer minimal viable experiments first
- Keep training/eval scripts simple and explicit
- Expose critical hyperparameters through arguments
- Make outputs reproducible and easy to review

## Execution Environment
- OS: Windows 11
- Shell: PowerShell
- If bash is available through Git Bash, prefer PowerShell-compatible commands unless explicitly needed
- Python env: sam2_privacy
- Activate: conda activate sam2_privacy
- Project root: E:\PycharmProjects\pythonProject\sam2_privacy_preprocessor
- ffmpeg: C:\ffmpeg\bin\ffmpeg.exe