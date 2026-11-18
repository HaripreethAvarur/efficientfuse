# efficientfuse (GreenFuse)

GreenFuse tests whether different ways of grouping GPU operations for compilation (TorchInductor), with or without CUDA Graph replay, reduce GPU energy while staying within 5% of the fastest runtime. The Python package and command are named `greenfuse`.

Results, numbers, and findings are in **[results/README.md](results/README.md)**. What was done in each version, and why, is in **[PROJECT_VERSION_LOG.md](PROJECT_VERSION_LOG.md)**.

## Repository structure

```
efficientfuse/
├── src/greenfuse/          the package
│   ├── cli.py              MAIN ENTRY POINT: the `greenfuse` command (inspect, calibrate, run, benchmark, select, report)
│   ├── workloads/          workload W1: relu(x + bias) + residual
│   ├── graph/, partition/  capture the op graph and enumerate groupings (p00 full, p01, p10, p11 separate)
│   ├── compiler/           compile each group with TorchInductor, read kernel evidence
│   ├── execution/          ordinary launch and CUDA Graph replay executors
│   ├── measurement/        runtime and NVML GPU energy measurement
│   └── planner/            select within the 5% runtime bound, validate, cache decisions
├── configs/                experiment configs (YAML)
├── scripts/
│   ├── check_environment.py, check_execution.py   setup checks
│   ├── export_experiment.py                       planner run -> readable summary.csv + plot
│   ├── exp3_screen_fanout.py                      Experiment 3 (standalone)
│   ├── exp3_inspect_kernels.py, exp3_trace_report.py   Experiment 3 kernel analysis
│   └── slurm/exp3_fanout_screen.sbatch            Slurm job for Experiment 3
├── tests/                  unit, integration, and GPU tests
├── results/                all measured results (see results/README.md)
├── docs/SETUP.md           detailed setup and command reference
└── PROJECT_VERSION_LOG.md  history: what changed in each version, findings, mistakes, reasons
```

## Reproduce

Needs a Linux machine with a whole NVIDIA GPU (results were measured on an A40) and gcc ≥ 9 (on Great Lakes: `module load gcc/13.2.0`).

**1. Install**

```bash
git clone https://github.com/HaripreethAvarur/efficientfuse.git && cd efficientfuse
conda create -n env_scalable python=3.11 -y && conda activate env_scalable
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[config,measurement,analysis,test]"
export CC=$(which gcc) CXX=$(which g++)
```

**2. Check the setup** (optional)

```bash
python scripts/check_environment.py --device cuda:0 --exclusive-gpu --output results/setup/environment.json
GREENFUSE_RUN_GPU_TESTS=1 GREENFUSE_EXCLUSIVE_GPU=1 python -m unittest discover -s tests
```

**3. W1, all groupings at all shapes** (covers Experiments 1 and 2)

```bash
greenfuse calibrate --config configs/all_groupings_w1_all_shapes.yaml --exclusive-gpu \
  --output-dir results/runs/all_groupings --cache-dir results/decisions

# Repeat for each shape; <run>/<id> are printed by calibrate
python scripts/export_experiment.py results/runs/all_groupings/<run>/w1_4096x4096/<id> \
  results/all_groupings_w1_all_shapes/4096x4096 --shape 4096x4096
```

Experiments 1 and 2 alone: use `configs/exp1_w1_small_last_pair_full.yaml` or `configs/exp2_w1_large_last_pair_full.yaml` in the same command.

**4. Experiment 3: fan-out workload, where partial fusion can win**

```bash
sbatch scripts/slurm/exp3_fanout_screen.sbatch     # set your own --account first, or run its python command directly
python scripts/exp3_inspect_kernels.py             # generated Triton code + per-kernel analysis
python scripts/exp3_trace_report.py --shape 8192x8192
```

Outputs go to `results/exp3_fanout_screen/`.
