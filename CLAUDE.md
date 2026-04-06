# Project Context

This project implements a SAM2 privacy preprocessor.

## Goals
- Turn IDEA_SUMMARY.md into runnable experiments
- Continue from the current scaffold instead of restarting from scratch
- Keep the implementation minimal and experiment-driven
- Save all experiment outputs to json/csv/log
- First determine the right number of V100 GPUs through profiling before scaling out

## Working Files
- Primary idea doc: IDEA_SUMMARY.md
- Experiment plan: refine-logs/EXPERIMENT_PLAN.md
- Experiment tracker: refine-logs/EXPERIMENT_TRACKER.md

## Local Control Environment
- OS: Windows 11
- Shell: PowerShell
- Project root: E:\PycharmProjects\pythonProject\sam2_privacy_preprocessor
- Local Python env: sam2_privacy_preprocessor
- Local activate command: conda activate sam2_privacy_preprocessor
- Local ffmpeg: C:\ffmpeg\bin\ffmpeg.exe

## Remote Server
- SSH alias: `lvshaoting-gpu` (configured in C:/Users/glitterrr/.ssh/config with key aris_ed25519)
- Connect: `ssh lvshaoting-gpu` (no password needed, key auth)
- Home directory: /IMBR_Data/Student-home/2025M_LvShaoting
- Code directory: /IMBR_Data/Student-home/2025M_LvShaoting/sam2_privacy_preprocessor
- GPUs: unknown (check with nvidia-smi after login)
- Primary target GPU type: Tesla V100
- Conda env: sam2_privacy_preprocessor
- Activate: eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate sam2_privacy_preprocessor
- Use screen or tmux for long-running jobs
- Assume Linux shell on the remote server

## Paths
- DAVIS_ROOT: /path/to/DAVIS
- CHECKPOINT_ROOT: /path/to/checkpoints
- Preferred checkpoint: /path/to/checkpoints/sam2.1_hiera_tiny.pt
- FFMPEG_PATH: ffmpeg

## Resource Policy
- Do not assume A100 or H100
- First run a 1-GPU profiling pilot
- Measure peak VRAM, step time, dataloader speed, ffmpeg overhead, and estimated wall-clock
- Only recommend 2 or 4 V100s if profiling shows clear speedup or parallel benefit
- Prefer fewer GPUs if run dependencies are sequential
- Distinguish between:
  - data parallelism
  - multi-experiment parallelism
- Recommend max_parallel_runs explicitly after profiling

## Implementation Rules
- Read IDEA_SUMMARY.md and refine-logs/EXPERIMENT_PLAN.md before changing code
- Modify the current scaffold instead of rebuilding a new framework
- Keep scripts simple, explicit, and argparse-driven
- Expose critical hyperparameters through arguments
- Save all results to json/csv/log
- Use conservative defaults first
- First round only supports:
  - B0 sanity
  - B1 baseline
  - minimal B2 main
- Do not implement B4/B6 in the first round unless explicitly requested

## Profiling First
- Before launching full experiments, create a 1-GPU profiling pilot
- Use profiling results to recommend whether 1, 2, or 4 V100 GPUs are best
- Report:
  - recommended GPU count
  - recommended batch size
  - recommended grad accumulation
  - recommended max_parallel_runs
  - reasons not to use more GPUs if scaling is not worthwhile
