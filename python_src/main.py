#!/usr/bin/env python3
"""Python port of src/main.rs (approximate).

Responsibilities:
- Parse CLI (expect a problem CSV path).
- Load environment from `.env`.
- Build a `Problem` from CSV and run:
    - heuristics: evaluate a few hand-coded routing+sequencing pairs.
    - GP: run genetic programming loop (population init, evaluate, select,
        crossover/mutate) and log results.

Key functions/classes:
- `heuristics(problem)`: runs simple routing+sequencing programs and logs
    heuristic results.
- `gp(problem)`: runs the GP loop using `GPContext` from `gp.mod`.
- `Individual`: small wrapper holding routing/sequencing `Program`s and
    evaluation helpers used by the GP loop.

Notes:
- Module expects the `gp`, `sim`, and `log` packages to be importable from
    `python_src` (package imports are used). Environment variables (or `.env`)
    control logging and GP parameters.

CLI Usage:
    python main.py <target_path> [num_instances] [num_scenarios]

    <target_path>   : thư mục gốc chứa các subfolder instance
                      (vd: datasets/h100_new)
    [num_instances] : số lượng instance cần chạy (mặc định: tất cả)
    [num_scenarios] : số lượng scenarios mỗi instance (mặc định: 16)

Dataset structure (sau khi reorganize):
    datasets/h100_new/
        h100c101/
            h100c101_1.csv
            h100c101_2.csv
            ...
        h100c102/
            h100c102_1.csv
            ...

    Mỗi subfolder = 1 instance. Tên subfolder = tên instance.
    Các file CSV bên trong = các scenarios của instance đó.
"""
import os
import sys
import re
import glob
import math
import random
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

dotenv_path = find_dotenv()
if not dotenv_path:
    alt = os.path.join(REPO_ROOT, '.env')
    if os.path.exists(alt):
        dotenv_path = alt
if dotenv_path:
    load_dotenv(dotenv_path)

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gp import mod as gp_mod
from gp import GPtree as gp_program
from log import logger as log_mod
from sim import mod as sim_mod
from sim import ctx as sim_ctx
from sim import problem as problem_mod
from sim.problem import ProblemSet

# Loggers
MAIN      = log_mod.Logger("MAIN")
HEU       = log_mod.Logger("HEU")
SIM       = log_mod.Logger("SIM")
GP        = log_mod.Logger("GP")
LASTPOP   = log_mod.Logger("LASTPOP")
LASTROUTE = log_mod.Logger("LASTROUTE")
ROUTE     = log_mod.Logger("ROUTE")
ROUTEEVAL = log_mod.Logger("ROUTEEVAL")
DEBUG     = log_mod.Logger("DEBUG")

# ── Configs ────────────────────────────────────────────────────────────────────
CONST_RATE           = float(os.environ.get("CONST_RATE",           "0.0"))
WEIGHT               = float(os.environ.get("WEIGHT",               "0.5"))
LATE_WEIGHT          = float(os.environ.get("LATE_WEIGHT",          "1"))
PENDING_WEIGHT       = float(os.environ.get("PENDING_WEIGHT",       "2"))

NUM_TIME_SLOT        = float(os.environ.get("NUM_TIME_SLOT",        "20.0"))
NUM_GEN              = int(os.environ.get("NUM_GEN",                "100"))
POP_SIZE             = int(os.environ.get("POP_SIZE",               "100"))
MAX_DEPTH            = int(os.environ.get("MAX_DEPTH",              "6"))
CROSSOVER_RATE       = float(os.environ.get("CROSSOVER_RATE",       "0.8"))
MUTATION_RATE        = float(os.environ.get("MUTATION_RATE",        "0.15"))
TRAIN_FACTOR         = float(os.environ.get("TRAIN_FACTOR",         "2.0"))
STRESS_FACTOR        = float(os.environ.get("STRESS_FACTOR",        "1.0"))
SCENARIO_TRAIN_RATIO = float(os.environ.get("SCENARIO_TRAIN_RATIO", "0.0"))


# ── Fitness ────────────────────────────────────────────────────────────────────
def fitness(problem, result):
    distance, profit = result
    tot_dist   = problem.truck_speed * problem.depot.close * float(problem.num_trucks)
    weight     = WEIGHT
    max_profit = sum(getattr(r, "profit", 0.0) for r in problem.requests)
    if max_profit == 0.0:
        max_profit = 1.0
    return distance / tot_dist * weight + ((max_profit - profit) / max_profit) * (1.0 - weight)


# ── Heuristics ─────────────────────────────────────────────────────────────────
def heuristics(problem_set: ProblemSet):
    try:
        CR  = sim_ctx.RoutingProgram.terminal(3)
    except Exception:
        CR  = gp_program.Program.terminal(3)
    try:
        CS  = sim_ctx.SequencingProgram.from_vec([
            gp_program.Node.Internal(5), gp_program.Node.Terminal(0), gp_program.Node.Terminal(4)])
    except Exception:
        CS  = gp_program.Program.from_vec([
            gp_program.Node.Internal(5), gp_program.Node.Terminal(0), gp_program.Node.Terminal(4)])
    try:
        W   = sim_ctx.SequencingProgram.terminal(3)
    except Exception:
        W   = gp_program.Program.terminal(3)
    try:
        WIQ = sim_ctx.RoutingProgram.terminal(1)
    except Exception:
        WIQ = gp_program.Program.terminal(1)
    try:
        PS  = sim_ctx.SequencingProgram.terminal(6)
    except Exception:
        PS  = gp_program.Program.terminal(6)

    time_slots = [prob.depot.close / NUM_TIME_SLOT for prob in problem_set]

    for name, r, s in [("C+C", CR, CS), ("C+W", CR, W), ("WIQ+C", WIQ, CS),
                       ("C+P", CR, PS), ("WIQ+P", WIQ, PS)]:
        results = []
        for prob, t_slot in zip(problem_set, time_slots):
            sim = sim_mod.Simulation(prob, r, s)
            dist, profit = sim.simulate_until(t_slot, float("inf"))
            results.append((dist, profit))

        avg_fit    = problem_set.update_fitness(results, fitness)
        avg_dist   = sum(r[0] for r in results) / len(results) if results else 0.0
        avg_profit = sum(r[1] for r in results) / len(results) if results else 0.0
        log_mod.log(HEU, "heuristic_result",
                    name=name, result=(avg_dist, avg_profit), fitness=avg_fit)


# ── Individual ─────────────────────────────────────────────────────────────────
class Individual:
    def __init__(self, routing, sequencing):
        self.routing    = routing
        self.sequencing = sequencing
        self.result     = None

    @staticmethod
    def ramp_half_and_half(gpc: gp_mod.GPContext):
        try:
            r_ctx = sim_ctx.RoutingContext
            s_ctx = sim_ctx.SequencingContext
        except Exception:
            r_ctx = None
            s_ctx = None
        r_pop = gpc.ramp_half_and_half(context=r_ctx)
        s_pop = gpc.ramp_half_and_half(context=s_ctx)
        return [Individual(r, s) for r, s in zip(r_pop, s_pop)]

    def crossover_with(self, gpc, other):
        r1, r2 = gpc.crossover(self.routing,    other.routing)
        s1, s2 = gpc.crossover(self.sequencing, other.sequencing)
        return Individual(r1, s1), Individual(r2, s2)

    def mutate(self, gpc):
        return Individual(gpc.mutation(self.routing), gpc.mutation(self.sequencing))

    def evaluate(self, cache, problem_set: ProblemSet, train_time_slots: List[float]):
        if self.result is not None:
            return self.result[2]
        cache_key = f"{self.routing}:{self.sequencing}"
        if cache_key in cache:
            res = cache[cache_key]
        else:
            results = []
            for prob, t_slot in zip(problem_set, train_time_slots):
                sim = sim_mod.Simulation(prob, self.routing, self.sequencing)
                dist, profit = sim.simulate_until(t_slot, float("inf"))
                results.append((dist, profit))
            fit        = problem_set.update_fitness(results, fitness)
            avg_dist   = sum(r[0] for r in results) / len(results) if results else 0.0
            avg_profit = sum(r[1] for r in results) / len(results) if results else 0.0
            res = (avg_dist, avg_profit, fit)
            cache[cache_key] = res
        self.result = res
        return res[2]


def select_parent(gpc, pop):
    idxs = random.sample(range(len(pop)), k=min(8, len(pop)))
    best = max(idxs, key=lambda i: pop[i].result[2])
    return best


# ── GP loop ────────────────────────────────────────────────────────────────────
def gp(problem_set: ProblemSet):
    time_slots       = [prob.depot.close / NUM_TIME_SLOT for prob in problem_set]
    train_time_slots = [t / STRESS_FACTOR for t in time_slots]
    limit            = math.ceil(len(problem_set) * SCENARIO_TRAIN_RATIO)

    training_problems = [
        prob.clone_training(t_slot * TRAIN_FACTOR, STRESS_FACTOR)
        for prob, t_slot in zip(problem_set[:limit], time_slots[:limit])
    ]
    training_problem_set = ProblemSet(training_problems)

    seed_env = os.environ.get("SEED", "")
    seed_val = None
    if seed_env:
        try:
            seed_val = int(seed_env)
        except Exception:
            pass

    rng = random.Random(seed_val) if seed_val is not None else random.Random()
    gpc = gp_mod.GPContext(rng=rng, num_population=POP_SIZE, max_depth=MAX_DEPTH)

    cache: dict = {}
    pop: List[Individual] = Individual.ramp_half_and_half(gpc)

    for gen in range(1, NUM_GEN + 1):
        for ind in pop:
            ind.evaluate(cache, training_problem_set, train_time_slots)

        pop.sort(key=lambda i: i.result[2])
        pop    = pop[:gpc.num_population]
        result = pop[0].result

        log_mod.log(GP, "new_gen",
                    gen=gen, result=(result[0], result[1]), fitness=result[2],
                    routing=str(pop[0].routing), sequencing=str(pop[0].sequencing))

        full_results = []
        for prob, t_slot in zip(problem_set, time_slots):
            sim_best = sim_mod.Simulation(prob, pop[0].routing, pop[0].sequencing)
            dist, profit = sim_best.simulate_until(t_slot, float("inf"))
            full_results.append((dist, profit))

        avg_fit    = problem_set.update_fitness(full_results, fitness)
        avg_dist   = sum(r[0] for r in full_results) / len(full_results) if full_results else 0.0
        avg_profit = sum(r[1] for r in full_results) / len(full_results) if full_results else 0.0
        log_mod.log(GP, "full_result",
                    result=(avg_dist, avg_profit), fitness=avg_fit)

        try:
            log_mod.log(GP, "base64",
                        routing=pop[0].routing.base64(),
                        sequencing=pop[0].sequencing.base64())
        except Exception:
            pass

        if gen == NUM_GEN:
            sim = sim_mod.Simulation(problem_set[0], pop[0].routing, pop[0].sequencing)
            sim.simulate_until(time_slots[0], float("inf"))
            for vehicle in range(problem_set[0].num_trucks):
                v = sim.vehicles[vehicle]
                log_mod.log(LASTROUTE, "route_log",
                            vehicle=vehicle,
                            route=getattr(v, "route",   None),
                            dropped=getattr(v, "dropped", None))
            for ind in pop:
                log_mod.log(LASTPOP, "lastpop",
                            routing=str(ind.routing),
                            sequencing=str(ind.sequencing))

        new_pop = list(pop)
        half    = gpc.num_population // 2
        for _ in range(half):
            p1 = select_parent(gpc, pop)
            p2 = select_parent(gpc, pop)
            x  = random.random()
            if x <= CROSSOVER_RATE:
                c1, c2 = pop[p1].crossover_with(gpc, pop[p2])
                new_pop.extend([c1, c2])
            elif x <= CROSSOVER_RATE + MUTATION_RATE:
                new_pop.extend([pop[p1].mutate(gpc), pop[p2].mutate(gpc)])
            else:
                new_pop.append(pop[p1])
                new_pop.append(pop[p2])
        pop = new_pop


# ── Dataset discovery ──────────────────────────────────────────────────────────
def discover_instances(target_dir: str, max_instances: int = None,
                       max_scenarios: int = 16) -> Dict[str, List[str]]:
    """
    Duyệt <target_dir> và trả về dict { instance_name: [csv_paths] }.

    Hỗ trợ 2 layout:

    Layout mới (subfolder):
        target_dir/
            h100c101/
                h100c101_1.csv
                h100c101_2.csv
            h100c102/
                h100c102_1.csv

    Layout cũ (flat):
        target_dir/
            h100c101_1.csv
            h100c101_2.csv
            h100c102_1.csv

    Với layout mới, tên subfolder = tên instance.
    Với layout cũ, tên instance được suy ra từ tên file (phần trước _<số>).
    """
    scenario_pattern = re.compile(r'^(.+)_(\d+)\.csv$', re.IGNORECASE)

    instances: Dict[str, List[str]] = {}

    # ── Layout mới: tìm các subfolder chứa CSV ─────────────────────────────
    subdirs = sorted([
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d))
    ])

    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(target_dir, subdir)
            csv_files   = sorted(
                glob.glob(os.path.join(subdir_path, "*.csv")),
                key=lambda p: _scenario_num(p, scenario_pattern)
            )
            if csv_files:
                instances[subdir] = csv_files
    else:
        # ── Layout cũ: flat CSV files ───────────────────────────────────────
        from collections import defaultdict
        groups: Dict[str, List[str]] = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(target_dir, "*.csv"))):
            basename = os.path.basename(f)
            m        = scenario_pattern.match(basename)
            inst     = m.group(1) if m else os.path.splitext(basename)[0]
            groups[inst].append(f)
        for inst, files in sorted(groups.items()):
            instances[inst] = sorted(
                files, key=lambda p: _scenario_num(p, scenario_pattern)
            )

    # ── Cắt giới hạn instance và scenario ─────────────────────────────────
    instance_names = sorted(instances.keys())
    if max_instances is not None:
        instance_names = instance_names[:max_instances]

    return {
        name: instances[name][:max_scenarios]
        for name in instance_names
    }


def _scenario_num(path: str, pattern: re.Pattern) -> int:
    m = pattern.match(os.path.basename(path))
    return int(m.group(2)) if m else 0


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """
    Usage:
        python main.py <target_path> [num_instances] [num_scenarios]

    <target_path>   : thư mục gốc chứa các subfolder instance
                      (vd: datasets/h100_new)
    [num_instances] : số instance tối đa cần chạy  (mặc định: tất cả)
    [num_scenarios] : số scenarios tối đa mỗi instance (mặc định: 16)

    Cấu trúc dataset mới:
        datasets/h100_new/
            h100c101/
                h100c101_1.csv
                h100c101_2.csv
                ...
            h100c102/
                ...
    """
    if len(sys.argv) < 2:
        print("usage: python main.py <target_path> [num_instances] [num_scenarios]")
        print()
        print("  <target_path>   : thư mục gốc chứa các subfolder instance")
        print("                    (vd: datasets/h100_new)")
        print("  [num_instances] : số instance tối đa cần chạy  (mặc định: tất cả)")
        print("  [num_scenarios] : số scenarios/instance         (mặc định: 16)")
        sys.exit(1)

    target        = sys.argv[1]
    max_instances = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    max_scenarios = int(sys.argv[3]) if len(sys.argv) >= 4 else 16

    if not os.path.isdir(target):
        print(f"[ERROR] Không tìm thấy thư mục: '{target}'")
        sys.exit(1)

    # ── Xác định num_trucks và truck_capacity theo kích thước bộ dữ liệu ──
    _DATASET_CFG = {
        100: (10,  200),
        200: (50,  400),
        400: (100, 800),
    }
    _target_str = target.replace("\\", "/")
    _dataset_size = next(
        (s for s in (400, 200, 100) if str(s) in _target_str.split("/")),
        None
    )
    if _dataset_size is None:
        print("[WARNING] Không nhận diện được kích thước bộ dữ liệu từ đường dẫn. "
              "Dùng mặc định: num_trucks=10, truck_capacity=200")
        _num_trucks, _truck_capacity = 10, 200
    else:
        _num_trucks, _truck_capacity = _DATASET_CFG[_dataset_size]
        print(f"Bộ dữ liệu {_dataset_size}: num_trucks={_num_trucks}, truck_capacity={_truck_capacity}")

    # ── Discover instances ─────────────────────────────────────────────────
    instances = discover_instances(target, max_instances, max_scenarios)

    if not instances:
        print(f"Không tìm thấy instance nào trong '{target}'")
        sys.exit(1)

    print(f"Tìm thấy {len(instances)} instance(s) sẽ chạy.")
    log_mod.log(MAIN, "start",
                total_instances=len(instances),
                max_scenarios=max_scenarios)

    # ── Chạy từng instance ─────────────────────────────────────────────────
    for inst_name, scenario_files in instances.items():
        print(f"\n{'='*60}")
        print(f"Instance: {inst_name}  ({len(scenario_files)} scenarios)")
        for f in scenario_files:
            print(f"  {f}")

        problem_set = ProblemSet.load_from_csvs(
            scenario_files,
            truck_speed=1.0,
            truck_capacity=float(_truck_capacity),
            num_trucks=_num_trucks,
        )

        log_mod.log(MAIN, "instance_start",
                    instance=inst_name,
                    scenarios=len(scenario_files))

        if HEU.enabled():
            log_mod.log(MAIN, "heu_start", instance=inst_name)
            heuristics(problem_set)

        if GP.enabled():
            log_mod.log(MAIN, "gp_start", instance=inst_name)
            gp(problem_set)

        log_mod.log(MAIN, "instance_done", instance=inst_name)

    log_mod.log(MAIN, "all_done", instances_run=len(instances))
    print("\nAll instances done.")


if __name__ == "__main__":
    main()