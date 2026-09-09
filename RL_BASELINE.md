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

## Fair evaluation budget

By default the RL search budget is:

```text
MAX_EVALUATIONS (if set), otherwise POP_SIZE * NUM_GEN
```

With the repository defaults this is `100 * 100 = 10,000` objective
evaluations. One objective evaluation means one sampled policy evaluated on
the same GP training subset,
`ceil(SCENARIO_TRAIN_RATIO * number_of_scenarios)`, then averaged. The JSON
records both:

- `objective_evaluations`: strict search budget, comparable to candidate GP
  evaluations / ACO ants;
- `training_simulator_rollouts`: physical simulator calls, equal to objective
  evaluations times the number of training scenarios.

Deterministic reporting over every requested scenario is stored separately as
`report_simulator_rollouts` and does not consume the search budget, analogous
to GP's `full_result` reporting pass.

Note that GP memoizes duplicate tree pairs, so its actual physical simulator
rollouts can be lower than `POP_SIZE * NUM_GEN`. The original GP selection,
RNG, and stopping behavior have intentionally not been changed by this RL
baseline.

## Running

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

The command writes `rl_results.json` beside the target dataset directory and
one `.pt` checkpoint per instance under `rl_checkpoints/` by default.

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

The audit also found pre-existing GP inconsistencies that were deliberately
left behavior-compatible: CSV GP/demo tournament selection uses `max` although
fitness and survivor selection use minimization; `tensor_main.py` uses `min`;
and `CONST_RATE` currently falls back to `0.1` in `gp/mod.py` under the normal
top-level import path. These should be resolved as a separate GP change so
historical baseline results are not silently invalidated.
