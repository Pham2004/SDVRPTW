# Masked PPO baseline for SDVRPTW

## Repository model used by the baseline

The CSV loader treats the first data row as the depot and all following rows
as customer requests. Columns are:

| Column | Meaning |
|---|---|
| `x`, `y` | Euclidean location |
| `demand` | Vehicle capacity consumed |
| `open`, `close` | Hard service-start time window |
| `servicetime` | Service duration |
| `drone_serve` | Present in the data but unused by the vehicle-only simulator |
| `time` | Dynamic reveal/arrival time |
| `profit` | Reward collected when served |
| `type` | Request type; loaded but not used by the current simulator |

The objective reused from `python_src/main.py` is minimized:

```text
fitness = WEIGHT * distance / (speed * depot_close * num_trucks)
        + (1-WEIGHT) * (max_profit - served_profit) / max_profit
```

The canonical `Simulation` batches a request into
`ceil(request.time / time_slot)`, assigns it to a feasible vehicle queue, and
selects the next queued request whenever that vehicle is idle. A vehicle
returns to the depot to refill when its remaining capacity is insufficient.

## RL algorithm

`python_src/rl_dvrptw.py` implements masked PPO without Gym or external RL
libraries. It retains the same two decision points as GP:

1. **Routing:** choose one time-feasible vehicle for a newly revealed request.
2. **Sequencing:** choose the next request from that vehicle's current queue.

A shared candidate-scoring network handles variable numbers of vehicles and
queued requests. Its 19 normalized features cover decision type, queue/load,
vehicle position and availability, travel, demand, time window, reveal time,
profit, slack, and waiting time. Invalid actions never enter the candidate set.

The step reward is an exact additive form of the GP objective (up to the
constant `1-WEIGHT`), so maximizing episode return ranks solutions exactly as
minimizing `main.fitness`.

## Fair one-scenario-per-instance protocol

The corrected GP and RL runners use the same experimental unit:

- one separately trained model per benchmark instance;
- the first CSV scenario is the only training scenario for that model;
- both methods use that CSV exactly, without GP's legacy `clone_training`
  transformation or RL bootstrap/jitter;
- the requested scenarios are evaluated deterministically after training;
- training and reporting simulator calls are counted separately.

RL repeats stochastic policy rollouts on that one fixed scenario. It no longer
bootstraps training data from the full target benchmark, so evaluation rows do
not feed the training-data generator. GP uses a seeded minimization tournament;
duplicate tree pairs remain memoized, and the manifest records its actual cache
misses rather than assuming every configured candidate caused a rollout.

With the defaults, each instance gets at most `100 * 100 = 10,000` GP objective
evaluations and exactly `10,000` RL training rollouts. Aggregate budget is the
per-instance value multiplied by the number of instances.

## One-command runs

### Corrected GP

```powershell
python python_src/run_gp_pipeline.py datasets/h200_new --seed 42 `
  --routing-depth 8 --sequencing-depth 6
python python_src/run_gp_pipeline.py datasets/h400_new --seed 42 `
  --routing-depth 8 --sequencing-depth 6
```

The runner defaults to routing depth 8 and sequencing depth 6. `--max-depth`
is retained only as a legacy shared override. Add `--use-training-clone` only
when reproducing the old GP data-transformation protocol rather than the fair
exact-CSV run.

### RL trained on exactly one scenario per instance

```powershell
# H100
python python_src/run_rl_pipeline.py datasets/h100_new --device cuda

# H200
python python_src/run_rl_pipeline.py datasets/h200_new --device cuda

# H400
python python_src/run_rl_pipeline.py datasets/h400_new --device cuda
```

Each invocation trains one checkpoint per discovered instance. For each model,
the dataset artifact has shape `(1, customers + 1, 10)` and contains the first
CSV scenario without bootstrap, jitter, or row resampling. Evaluation uses up
to 16 scenarios from the same instance. The timestamped output directory has
aggregate `results.json`, `summary.csv`, and `run_manifest.json`, plus one
subdirectory per instance containing its dataset, checkpoint, and raw results.

The fleet mapping remains H100 `10/200`, H200 `50/400`, and H400 `100/800`.

## Full GP/RL benchmark suite

Run both algorithms on H100, H200, and H400 with paired seeds 42--46:

```powershell
.\run_all_experiments.bat
```

The default protocol trains one model/policy per instance on scenario 1 and
reports only scenarios 2--16. GP uses routing depth 8, sequencing depth 6, and
a nominal 10,000 evaluations (100 generations x 100 population); RL uses
10,000 evaluations. Results are written under `benchmark_runs/gp_rl_5seeds/`.
Rerunning the same command resumes the suite and skips completed runs.

The main aggregate files are `runs.csv` (every seed),
`summary_by_algorithm.csv` (mean/std/95% CI and actual budgets),
`summary_by_instance.csv`, `paired_comparison.csv`, and
`comparison_summary.csv`. Fitness is minimized; the paired delta is
`RL fitness - GP fitness`, so a positive value means GP won that seed.

To force RL onto a specific GPU, pass for example `--device cuda:0`. Use a new
`--output-dir` whenever changing benchmark parameters.
Useful overrides are:

```powershell
python python_src/run_rl_pipeline.py datasets/h400_new --device cuda:0 `
  --max-evaluations 10000 --max-scenarios 16 --seed 43
```

To run the three stages manually for one instance:

```powershell
# Exact scenario; the resulting tensor contains one problem only.
python python_src/generate_rl_dataset.py `
  datasets/h200_new/h200c101/h200c101_1.csv `
  datasets/rl_train_h200c101.pt

python python_src/train_rl_checkpoint.py datasets/rl_train_h200c101.pt `
  --output rl_checkpoints/h200c101_seed42.pt `
  --max-evaluations 10000 --seed 42 --device cuda

python python_src/evaluate_rl_checkpoint.py `
  rl_checkpoints/h200c101_seed42.pt datasets/h200_new/h200c101 `
  --output datasets/h200c101_rl_seed42.json --device cuda
```

Passing a directory to `generate_rl_dataset.py` still exposes the old synthetic
bootstrap utility for controlled ablations, but the fair pipeline never uses
that mode.

For a paper, repeat both complete GP and RL pipelines for the same independent
seeds (for example `42` through `51`) and report mean/std over corresponding
per-instance results.

## Legacy per-instance command

Install requirements, then run for example:

```powershell
python -m pip install -r requirements.txt
python python_src/rl_dvrptw.py datasets/h100_new 1 16
```

Useful overrides:

```powershell
python python_src/rl_dvrptw.py datasets/h100_new 1 16 `
  --max-evaluations 10000 `
  --evaluations-per-update 32 `
  --ppo-epochs 4 `
  --seed 42 `
  --device cpu
```

This older command writes `rl_results.json` beside the target dataset directory
and trains checkpoints under `rl_checkpoints/`. Keep it only for reproducing
the first baseline; use `run_rl_pipeline.py` for auditable fair runs.

For a fast end-to-end check:

```powershell
python python_src/rl_dvrptw.py datasets/toy5 1 3 `
  --max-evaluations 8 --evaluations-per-update 4 `
  --ppo-epochs 1 --hidden-dim 32 --device cpu
```

## Tests and simulator audit

Run:

```powershell
python -m unittest discover -s python_src/tests -v
```

The regression suite checks hand-computed distance/profit, reveal-slot
rounding, duplicate visit timestamps, oversized requests, invalid vehicle
configuration, single-use simulation, objective-equivalent RL reward, and a
strict non-divisible RL budget.

The audit fixed these canonical simulator defects:

- CSV simulation no longer imports PyTorch eagerly;
- vehicle feasibility includes `busy_until`;
- oversized requests are dropped instead of creating a zero-time depot loop;
- equal-time visits no longer overwrite each other in the route dictionary;
- invalid speed/truck configuration and accidental simulator reuse fail fast;
- dropped requests are exposed through `Simulation.dropped_requests`.

The GP audit additionally fixed reversed tournament selection, routed parent
selection and crossover decisions through the configured seeded RNG, removed
the circular-import fallback that silently changed `CONST_RATE`, and added
regression coverage for minimization selection.
