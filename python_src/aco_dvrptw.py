"""
DACO – Dynamic Ant Colony Optimisation for DVRPTW
===================================================
Adapted from dvrpsd-main/algorithms/td_daco.py to run on the
GP-DVRPTW stochastic model (h100_new datasets).

Tích hợp môi trường có sẵn trong GP-DVRPTW-main:
  - sim.problem.Problem / ProblemSet / Request  (không tự định nghĩa lại)
  - Hàm fitness, discover_instances, _DATASET_CFG từ main.py
  - CLI giống hệt main.py

Key differences vs original td_daco (dvrpsd-main):
  1. Fully dynamic  – không có static phase; mọi request xử lý khi đến.
  2. Vehicle-only   – xóa toàn bộ drone; chỉ dùng xe.
  3. GP objective   – cùng hàm fitness với main.py.
  4. Stochastic     – mỗi ant đánh giá trên TẤT CẢ 16 scenarios; fitness = trung bình.
  5. Budget-limited – tổng evaluations giới hạn bằng GP budget
                      (POP_SIZE × NUM_GEN = 100 × 100 = 10 000).

Real-time dispatch model (giống dvrpsd-main):
  - Mỗi time slot commit dispatch decisions vào committed_state.
  - Slot tiếp theo xây plan TRÊN TOP committed_state đó (xe đang đi không thể assign lại).
  - Mỗi ant evaluate bằng simulate_from_state() bắt đầu từ committed_state hiện tại.
  - Sau mỗi slot: iteration-best ant's decisions được commit (locked).

Usage
-----
    python aco_dvrptw.py <target_path> [num_instances] [num_scenarios]

    <target_path>   : thư mục gốc chứa các subfolder instance
                      (vd: datasets/h100_new)
    [num_instances] : số instance tối đa cần chạy  (mặc định: tất cả)
    [num_scenarios] : số scenarios/instance         (mặc định: 16)
"""

import os
import sys
import csv
import copy
import json
import random
import time as _time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────
# JSON-line logger (giống log_mod trong main.py)
# Ghi vào file được chỉ định bởi LOG_ACO env var.
# Nếu không set LOG_ACO thì bỏ qua (không ghi).
# ──────────────────────────────────────────────────────────────────

_LOG_ACO_FILE: Optional[str] = os.environ.get("LOG_ACO")


def _log(logger_name: str, event: str, **kwargs) -> None:
    """Ghi một dòng JSON vào LOG_ACO file (nếu được cấu hình)."""
    if not _LOG_ACO_FILE:
        return
    entry = {"__": logger_name, "_": event}
    entry.update(kwargs)
    with open(_LOG_ACO_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────
# Tích hợp môi trường GP-DVRPTW-main
# ──────────────────────────────────────────────────────────────────

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.problem import Problem, ProblemSet, Request
from main import discover_instances, _scenario_num, fitness as _gp_fitness


# ──────────────────────────────────────────────────────────────────
# Fitness wrapper
# ──────────────────────────────────────────────────────────────────

def fitness(problem: Problem, result) -> float:
    return _gp_fitness(problem, result)


# ──────────────────────────────────────────────────────────────────
# Distance matrix helper
# ──────────────────────────────────────────────────────────────────

def build_dist_matrix(problem: Problem) -> np.ndarray:
    """Tính ma trận khoảng cách Euclidean; hàng/cột 0 = depot."""
    all_nodes = [problem.depot] + problem.requests
    xs = np.array([nd.x for nd in all_nodes])
    ys = np.array([nd.y for nd in all_nodes])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    return np.sqrt(dx * dx + dy * dy)


# ──────────────────────────────────────────────────────────────────
# Committed state helpers
# Giống dvrpsd-main: present_route = phần đã commit (xe đang chạy)
# ──────────────────────────────────────────────────────────────────

NUM_TIMESLOTS: int = 20


def _init_committed(num_v: int, cap0: float) -> dict:
    """Khởi tạo committed state rỗng tại t=0 (xe ở depot, chưa serve gì)."""
    return {
        'v_pos':        [0]    * num_v,   # matrix position (0=depot)
        'v_time':       [0.0]  * num_v,   # thời điểm xe hoàn thành TẤT CẢ đơn trong queue
        'v_cap':        [cap0] * num_v,   # capacity còn lại
        'v_queue':      [[]    for _ in range(num_v)],  # queue đơn hàng (idx) của mỗi xe
        'served':       set(),            # request.idx đã phục vụ
        'total_dist':   0.0,
        'total_profit': 0.0,
    }


def _commit_slot(
    priority_order: List[int],
    committed:      dict,
    problem:        Problem,
    dist_matrix:    np.ndarray,
    current_time:   float,
) -> dict:
    """
    Commit dispatch decisions tại current_time (1 bước thời gian).

      - Xe ĐANG RẢNH (v_time[v] <= current_time): khởi hành ngay tại current_time.
      - Xe ĐANG BẬN  (v_time[v] >  current_time): khởi hành sau khi hoàn thành
        đơn hàng hiện tại, tức tại v_time[v].  Dispatch được lên kế hoạch ngay
        tại slot này và lock vào committed state.
      - Kết quả được 'lock' → slot sau không thể thay đổi.

    Returns
    -------
    New committed state sau dispatch tại current_time.
    """
    D    = dist_matrix
    spd  = problem.truck_speed
    nv   = problem.num_trucks
    INF  = float('inf')

    all_nodes = [problem.depot] + problem.requests
    idx2pos: Dict[int, int] = {nd.idx: i for i, nd in enumerate(all_nodes)}
    prio:    Dict[int, int] = {rid: i for i, rid in enumerate(priority_order)}

    v_pos   = list(committed['v_pos'])
    v_time  = list(committed['v_time'])
    v_cap   = list(committed['v_cap'])
    v_queue = [list(q) for q in committed['v_queue']]
    served: Set[int] = set(committed['served'])
    total_dist   = committed['total_dist']
    total_profit = committed['total_profit']

    available: List[Request] = sorted(
        (r for r in problem.requests
         if r.time <= current_time and r.idx not in served),
        key=lambda r: prio.get(r.idx, 99_999),
    )

    for req in available:
        pos    = idx2pos[req.idx]
        d_req  = req.demand
        cl_req = req.close
        op_req = req.open

        best_v      = -1
        best_cost   = INF

        for v in range(nv):
            if d_req > v_cap[v]:
                continue
            # Xe rảnh khởi hành ngay; xe bận khởi hành sau khi hoàn thành toàn bộ queue
            depart = max(v_time[v], current_time)
            dist_d = D[v_pos[v], pos]
            arr    = depart + dist_d / spd
            if arr <= cl_req and dist_d - req.profit < best_cost:
                best_cost = dist_d - req.profit
                best_v    = v

        if best_v == -1:
            continue   # không có xe phù hợp → request bị bỏ qua slot này

        # Thêm đơn vào cuối queue của xe best_v; v_time cập nhật = thời điểm hoàn thành hết queue
        depart          = max(v_time[best_v], current_time)
        dist_d          = D[v_pos[best_v], pos]
        arr             = depart + dist_d / spd
        end             = max(arr, op_req) + req.service_time
        total_dist     += dist_d
        v_time[best_v]  = end           # thời điểm xe rảnh sau khi phục vụ hết queue
        v_cap[best_v]  -= d_req
        v_pos[best_v]   = pos           # vị trí cuối cùng trong queue
        v_queue[best_v].append(req.idx) # thêm vào cuối queue
        total_profit   += req.profit
        served.add(req.idx)

    return {
        'v_pos':        v_pos,
        'v_time':       v_time,
        'v_cap':        v_cap,
        'v_queue':      v_queue,
        'served':       served,
        'total_dist':   total_dist,
        'total_profit': total_profit,
    }


def simulate_from_state(
    priority_order: List[int],
    committed:      dict,
    problem:        Problem,
    dist_matrix:    np.ndarray,
    start_time:     float,
    timestep:       float,
) -> Tuple[float, float]:
    """
    Real-time dispatch simulation từ committed state đến hết scenario.

    Giống dvrpsd-main: mỗi ant nhận present_route (committed) rồi
    xây plan cho các request còn lại. Simulate từ start_time đến
    max_arrival, trả về full fitness (committed + new).

    Rules:
      - Mọi xe đều có thể nhận thêm đơn (xe bận thêm vào cuối queue).
      - Xe rảnh khởi hành ngay tại current_time; xe bận khởi hành tại v_time[v].
      - Request chưa served được thử lại ở các timestep sau.

    Parameters
    ----------
    priority_order : thứ tự ưu tiên cho các visible request chưa served.
    committed      : trạng thái locked từ các slot trước.
    start_time     : bắt đầu simulate từ đây (= current slot time).

    Returns
    -------
    (total_distance, total_profit) tính từ đầu (bao gồm committed portion).
    """
    if not problem.requests:
        d = committed['total_dist']
        for v in range(problem.num_trucks):
            d += dist_matrix[committed['v_pos'][v], 0]
        return d, committed['total_profit']

    D    = dist_matrix
    spd  = problem.truck_speed
    nv   = problem.num_trucks
    INF  = float('inf')

    all_nodes = [problem.depot] + problem.requests
    idx2pos: Dict[int, int] = {nd.idx: i for i, nd in enumerate(all_nodes)}
    prio:    Dict[int, int] = {rid: i for i, rid in enumerate(priority_order)}

    v_pos   = list(committed['v_pos'])
    v_time  = list(committed['v_time'])
    v_cap   = list(committed['v_cap'])
    v_queue = [list(q) for q in committed['v_queue']]
    served: Set[int] = set(committed['served'])
    total_dist   = committed['total_dist']
    total_profit = committed['total_profit']

    max_arrival  = max(r.time for r in problem.requests)
    current_time = start_time

    while current_time <= max_arrival + timestep:
        available: List[Request] = sorted(
            (r for r in problem.requests
             if r.time <= current_time and r.idx not in served),
            key=lambda r: prio.get(r.idx, 99_999),
        )
        for req in available:
            pos    = idx2pos[req.idx]
            d_req  = req.demand
            cl_req = req.close
            op_req = req.open

            best_v    = -1
            best_cost = INF

            for v in range(nv):
                if d_req > v_cap[v]:
                    continue
                # Xe rảnh khởi hành ngay; xe bận (đang có queue) khởi hành sau khi hết queue
                depart = max(v_time[v], current_time)
                dist_d = D[v_pos[v], pos]
                arr    = depart + dist_d / spd
                if arr <= cl_req and dist_d - req.profit < best_cost:
                    best_cost = dist_d - req.profit
                    best_v    = v

            if best_v == -1:
                continue

            # Thêm đơn vào cuối queue; v_time = thời điểm hoàn thành toàn bộ queue mới
            depart          = max(v_time[best_v], current_time)
            dist_d          = D[v_pos[best_v], pos]
            arr             = depart + dist_d / spd
            end             = max(arr, op_req) + req.service_time
            total_dist     += dist_d
            v_time[best_v]  = end
            v_cap[best_v]  -= d_req
            v_pos[best_v]   = pos
            v_queue[best_v].append(req.idx)
            total_profit   += req.profit
            served.add(req.idx)

        current_time += timestep

    # Trả tất cả xe về depot
    for v in range(nv):
        total_dist += D[v_pos[v], 0]

    return total_dist, total_profit


# ──────────────────────────────────────────────────────────────────
# Dynamic ACO
# ──────────────────────────────────────────────────────────────────

class DynamicACO:
    """
    Dynamic ACO cho stochastic DVRPTW.

    Kiến trúc real-time dispatch (giống dvrpsd-main):
      - Mỗi slot k: committed_state chứa dispatches đã lock từ slot 1..k-1.
      - Mỗi ant: simulate_from_state() xây plan TRÊN TOP committed_state.
      - Sau slot: iteration-best ant → _commit_slot() → cập nhật committed_state.
      - Slot tiếp theo bắt đầu từ committed_state mới.

    Stochastic:
      - 16 scenarios, mỗi scenario có committed_state riêng.
      - Fitness = trung bình qua 16 scenarios.

    Budget matching GP:
      - GP: POP_SIZE × NUM_GEN = 100 × 100 = 10 000 evaluations.
      - ACO: MAX_EVALUATIONS = 10 000 (cùng budget).
    """

    def __init__(
        self,
        problem_set:     ProblemSet,
        max_evaluations: int   = 10_000,
        num_ants:        int   = 50,
        alpha:           float = 1.0,
        beta:            float = 2.0,
        rho:             float = 0.6,
        q:               float = 0.05,
        timestep:        Optional[float] = None,
    ):
        if not problem_set.problems:
            raise ValueError("ProblemSet rỗng.")

        self.problem_set     = problem_set
        self.problems        = problem_set.problems
        self.max_evaluations = max_evaluations
        self.num_ants        = num_ants
        self.alpha           = alpha
        self.beta            = beta
        self.rho             = rho
        self.q               = q

        _max_arrival_init = max(
            (r.time for prob in problem_set.problems for r in prob.requests),
            default=0.0,
        )
        self.timestep: float = (
            timestep if timestep is not None
            else max(_max_arrival_init / NUM_TIMESLOTS, 1.0)
        )

        self.ref   = self.problems[0]
        n_req      = len(self.ref.requests)
        n_nodes    = n_req + 1

        self.dist_matrices: List[np.ndarray] = [
            build_dist_matrix(p) for p in self.problems
        ]

        self.req_positions: List[int] = list(range(1, n_nodes))
        self.req_ids: List[int]       = [r.idx for r in self.ref.requests]

        self.etas: List[np.ndarray] = [
            self._build_eta(p, D)
            for p, D in zip(self.problems, self.dist_matrices)
        ]

        tau0           = 1.0
        self.pheromone = np.full((n_nodes, n_nodes), tau0)
        np.fill_diagonal(self.pheromone, 0.0)

        self.req2pos: Dict[int, int] = {r.idx: i + 1 for i, r in enumerate(self.ref.requests)}

        self.best_fitness:          float                    = float("inf")
        self.best_fitness_per_scen: List[float]              = [float("inf")] * len(self.problems)
        self.best_orders:           Optional[List[List[int]]] = None
        self.evaluation_count:      int                      = 0
        self.history:               List[float]              = []
        self.final_committed:       Optional[List[dict]]     = None

    # ── Heuristic builder ─────────────────────────────────────────

    def _build_eta(self, prob: Problem, D: np.ndarray) -> np.ndarray:
        n_nodes = len(prob.requests) + 1

        profit  = np.zeros(n_nodes)
        urgency = np.zeros(n_nodes)
        for i, req in enumerate(prob.requests):
            pos = i + 1
            profit[pos]  = max(req.profit, 0.0)
            urgency[pos] = 1.0 / max(req.close - req.open, 1e-9)

        if profit.max() > 0:
            profit  /= profit.max()
        if urgency.max() > 0:
            urgency /= urgency.max()

        node_score = (profit + 1e-3) * (urgency + 1e-3)
        eta = node_score[np.newaxis, :] / (D + 1e-9)
        np.fill_diagonal(eta, 0.0)
        return eta

    # ── Solution construction ──────────────────────────────────────

    def construct_solution(self, eta: np.ndarray,
                           positions: Optional[List[int]] = None) -> List[int]:
        """
        Roulette-wheel ACO xây dựng priority ordering.
        positions: visible-and-unserved request matrix positions.
        """
        unvisited  = list(positions if positions is not None else self.req_positions)
        order_pos: List[int] = []
        cur = 0

        while unvisited:
            tau      = self.pheromone[cur, unvisited]
            eta_row  = eta[cur, unvisited]
            scores   = (tau ** self.alpha) * (eta_row ** self.beta)
            total    = scores.sum()

            if total <= 0.0:
                chosen_idx = random.randrange(len(unvisited))
            else:
                r      = random.random() * total
                cumsum = np.cumsum(scores)
                idx_arr = np.searchsorted(cumsum, r)
                chosen_idx = int(min(idx_arr, len(unvisited) - 1))

            chosen = unvisited[chosen_idx]
            order_pos.append(chosen)
            unvisited.pop(chosen_idx)
            cur = chosen

        pos2req = {i + 1: r.idx for i, r in enumerate(self.ref.requests)}
        return [pos2req[p] for p in order_pos]

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(
        self,
        orders:           List[List[int]],
        committed_states: List[dict],
        start_time:       float,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate một ant trên tất cả scenarios, bắt đầu từ committed_states.

        Giống dvrpsd-main: solution = present_route + ant's new assignments.
        simulate_from_state() bắt đầu từ committed_state, simulate đến hết.

        Returns (avg_fitness, per_scenario_fitness).
        """
        results = []
        for order, prob, D, cs in zip(
                orders, self.problems, self.dist_matrices, committed_states):
            d_val, profit = simulate_from_state(
                order, cs, prob, D, start_time, self.timestep)
            results.append((d_val, profit))

        self.problem_set.update_fitness(results, fitness)
        per_fit = list(self.problem_set.fitnesses)
        avg_fit = sum(per_fit) / len(per_fit)
        self.evaluation_count += 1
        return avg_fit, per_fit

    # ── Pheromone update ──────────────────────────────────────────

    def update_pheromone(
        self, batch: List[Tuple[List[List[int]], List[float]]]
    ) -> None:
        """Elitist global update. Deposit từ iteration-best và global-best ant."""
        self.pheromone *= (1.0 - self.rho)

        ib_idx    = min(range(len(batch)), key=lambda i: sum(batch[i][1]) / len(batch[i][1]))
        ib_orders, ib_fits = batch[ib_idx]
        for order, fit in zip(ib_orders, ib_fits):
            delta = (self.q / fit) if fit > 0.0 else self.q
            pos   = [0] + self._to_positions(order) + [0]
            for i in range(len(pos) - 1):
                self.pheromone[pos[i], pos[i + 1]] += delta

        if self.best_orders is not None:
            for order, fit in zip(self.best_orders, self.best_fitness_per_scen):
                delta = (self.q / fit) if fit > 0.0 else self.q
                pos   = [0] + self._to_positions(order) + [0]
                for i in range(len(pos) - 1):
                    self.pheromone[pos[i], pos[i + 1]] += delta

    def _to_positions(self, order_idx: List[int]) -> List[int]:
        idx2pos = {r.idx: i + 1 for i, r in enumerate(self.ref.requests)}
        return [idx2pos[rid] for rid in order_idx]

    def _reset_pheromone(self) -> None:
        self.pheromone[:] = 1.0
        np.fill_diagonal(self.pheromone, 0.0)

    # ── Main loop ─────────────────────────────────────────────────

    def run(self) -> Tuple[float, Optional[List[List[int]]]]:
        """
        Dynamic ACO với outer loop theo time slot.

        Giống dvrpsd-main dynamic_routing():
          slot k:
            1. Visible-but-unserved requests per scenario.
            2. ACO: mỗi ant evaluate = simulate_from_state(committed, start=k*Δt).
            3. Update pheromone.
            4. Commit: iteration-best ant → _commit_slot() → lock dispatches tại k*Δt.
            5. committed_state carry-over sang slot k+1.
        """
        n_scen = len(self.problems)
        n_req  = len(self.ref.requests)
        cap0   = self.ref.truck_capacity
        nv     = self.ref.num_trucks

        # Khởi tạo committed state rỗng cho mỗi scenario (giống present_route = [] ở dvrpsd)
        committed_states: List[dict] = [
            _init_committed(nv, cap0) for _ in range(n_scen)
        ]

        slot_times     = [(i + 1) * self.timestep for i in range(NUM_TIMESLOTS)]
        evals_per_slot = max(self.num_ants, self.max_evaluations // NUM_TIMESLOTS)

        print(f"\n[ACO] scenarios={n_scen}  requests={n_req}  "
              f"trucks={nv}  budget={self.max_evaluations}")
        print(f"      timeslots={NUM_TIMESLOTS}  Δt={self.timestep:.1f}  "
              f"evals/slot={evals_per_slot}")

        for slot_idx, current_time in enumerate(slot_times):
            if self.evaluation_count >= self.max_evaluations:
                break

            # Visible-but-unserved requests tại slot này (per scenario)
            vis_per_scen: List[List[int]] = [
                [self.req2pos[r.idx]
                 for r in prob.requests
                 if r.time <= current_time
                    and r.idx not in committed_states[s]['served']]
                for s, prob in enumerate(self.problems)
            ]

            if not any(vis_per_scen):
                continue

            if slot_idx % 4 == 0:
                self._reset_pheromone()

            batch: List[Tuple[List[List[int]], List[float]]] = []
            slot_evals = 0

            while (slot_evals < evals_per_slot
                   and self.evaluation_count < self.max_evaluations):

                # ACO xây priority order chỉ trên visible-unserved requests
                orders = [
                    self.construct_solution(eta, vis_pos)
                    for eta, vis_pos in zip(self.etas, vis_per_scen)
                ]

                # Evaluate: simulate từ committed_state, bắt đầu tại current_time
                avg_fit, per_fit = self.evaluate(orders, committed_states, current_time)
                batch.append((orders, per_fit))
                slot_evals += 1

            if batch:
                self.update_pheromone(batch)

            # Commit: iteration-best ant → lock dispatches tại current_time
            # Giống dvrpsd-main: cập nhật present_route từ planning_route
            ib_idx = min(range(len(batch)),
                         key=lambda i: sum(batch[i][1]) / len(batch[i][1]))
            ib_orders = batch[ib_idx][0]
            for s, (order, prob, D) in enumerate(
                    zip(ib_orders, self.problems, self.dist_matrices)):
                committed_states[s] = _commit_slot(
                    order, committed_states[s], prob, D, current_time
                )

            n_vis    = len(vis_per_scen[0]) if vis_per_scen else 0
            n_served = len(committed_states[0]['served'])
            print(f"  slot={slot_idx+1:2d}/{NUM_TIMESLOTS}  "
                  f"t={current_time:.0f}  "
                  f"vis={n_vis}/{n_req}  committed={n_served}  "
                  f"evals={self.evaluation_count:6d}")

        # Lưu final committed state để main() dùng cho breakdown
        self.final_committed = committed_states

        # Tính fitness thực sự 1 lần từ committed_states cuối (sau khi dispatch hết)
        final_results = []
        for cs, prob, D in zip(committed_states, self.problems, self.dist_matrices):
            dist_with_return = cs['total_dist'] + sum(
                D[cs['v_pos'][v], 0] for v in range(prob.num_trucks)
            )
            final_results.append((dist_with_return, cs['total_profit']))

        self.problem_set.update_fitness(final_results, fitness)
        per_fit = list(self.problem_set.fitnesses)
        self.best_fitness          = sum(per_fit) / len(per_fit)
        self.best_fitness_per_scen = per_fit

        print(f"[ACO] Done.  evals={self.evaluation_count}  "
              f"best_fitness={self.best_fitness:.5f}")
        return self.best_fitness, self.best_orders


# ──────────────────────────────────────────────────────────────────
# Entry point – CLI giống hệt main.py
# ──────────────────────────────────────────────────────────────────

_DATASET_CFG = {
    100: (10,  200.0),
    200: (50,  400.0),
    400: (100, 800.0),
}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python aco_dvrptw.py <target_path> [num_instances] [num_scenarios]")
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

    TRUCK_SPEED = 1.0

    # ── Seed (giống main.py: đọc từ env var SEED) ───────────────────
    _seed_env = os.environ.get("SEED", "")
    _seed_val: Optional[int] = None
    if _seed_env:
        try:
            _seed_val = int(_seed_env)
        except ValueError:
            pass
    if _seed_val is not None:
        random.seed(_seed_val)
        np.random.seed(_seed_val)
        print(f" Seed            : {_seed_val}")
    else:
        print(" Seed            : (none – non-deterministic)")

    MAX_EVALUATIONS = 10_000   # fair vs GP: POP_SIZE(100) × NUM_GEN(100); mỗi eval = 1 fitness trung bình qua 16 scenarios
    NUM_ANTS        = 50
    ALPHA, BETA, RHO, Q = 1.0, 2.0, 0.6, 0.005

    instances = discover_instances(target, max_instances, max_scenarios)
    if not instances:
        print(f"Không tìm thấy instance nào trong '{target}'")
        sys.exit(1)

    # ── Suy ra num_trucks / truck_capacity từ dữ liệu thật ──────────
    # Đọc scenario csv đầu tiên, đếm số customer (= số dòng dữ liệu - depot).
    _first_csv = next(iter(instances.values()))[0]
    with open(_first_csv, newline='') as _fh:
        _n_rows = sum(1 for _r in csv.reader(_fh) if _r and any(c.strip() for c in _r))
    _n_customers = max(_n_rows - 2, 0)  # trừ header + depot

    # Chọn config gần nhất theo số customer
    _dataset_size = min(_DATASET_CFG.keys(),
                       key=lambda s: abs(s - _n_customers))
    NUM_TRUCKS, TRUCK_CAPACITY = _DATASET_CFG[_dataset_size]
    print(f"Phát hiện {_n_customers} customers → dùng config bộ {_dataset_size}: "
          f"num_trucks={NUM_TRUCKS}, truck_capacity={TRUCK_CAPACITY}")

    print("=" * 65)
    print(" Dynamic ACO for DVRPTW  (real-time dispatch, vehicle-only)")
    print("=" * 65)
    print(f" Dataset        : {target}")
    print(f" Budget/instance: {MAX_EVALUATIONS} evals "
          f"({NUM_ANTS} ants × ~{MAX_EVALUATIONS // NUM_ANTS} iter)")
    print(f" Scenarios/inst : {max_scenarios}")
    print(f" Trucks         : {NUM_TRUCKS}  cap={TRUCK_CAPACITY}  "
          f"speed={TRUCK_SPEED}")
    print(f" ACO params     : α={ALPHA} β={BETA} ρ={RHO} q={Q}")
    print("=" * 65)
    print(f"\nTìm thấy {len(instances)} instance(s) sẽ chạy.")

    all_results: Dict = {}

    _log("MAIN", "start", total_instances=len(instances))

    for inst_name, scen_files in instances.items():
        print(f"\n{'─'*65}")
        print(f" Instance: {inst_name}  ({len(scen_files)} scenarios)")
        print(f"{'─'*65}")

        # Emit instance_start (giống main.py / aggregate_results.py expects)
        _log("MAIN", "instance_start", instance=inst_name,
             scenarios=len(scen_files))

        problem_set = ProblemSet.load_from_csvs(
            scen_files,
            truck_speed=TRUCK_SPEED,
            truck_capacity=TRUCK_CAPACITY,
            num_trucks=NUM_TRUCKS,
        )

        t0  = _time.time()
        aco = DynamicACO(
            problem_set     = problem_set,
            max_evaluations = MAX_EVALUATIONS,
            num_ants        = NUM_ANTS,
            alpha           = ALPHA,
            beta            = BETA,
            rho             = RHO,
            q               = Q,
        )
        best_fit, _ = aco.run()
        elapsed = _time.time() - t0

        # ── Per-scenario breakdown từ final committed state ─────────
        # Giống dvrpsd-main: kết quả cuối = present_route sau tất cả slots
        print(f"\n Per-scenario results – {inst_name}:")
        per_scen = []
        for i, (prob, D) in enumerate(zip(problem_set.problems, aco.dist_matrices)):
            cs = aco.final_committed[i]
            # Cộng thêm depot return (chưa có trong committed total_dist)
            depot_return = sum(D[cs['v_pos'][v], 0] for v in range(prob.num_trucks))
            d_val  = cs['total_dist'] + depot_return
            profit = cs['total_profit']
            f_val  = fitness(prob, (d_val, profit))
            per_scen.append({
                "scenario": i + 1,
                "distance": round(d_val,  4),
                "profit":   round(profit, 4),
                "fitness":  round(f_val,  5),
                "served":   len(cs['served']),
            })
            print(f"  scen {i+1:2d}: dist={d_val:8.2f}  "
                  f"profit={profit:7.2f}  fitness={f_val:.5f}  "
                  f"served={len(cs['served'])}")
            # Emit per-scenario log (aggregate_aco_results.py đọc)
            _log("ACO", "scenario_result",
                 scenario=i + 1,
                 result=[round(d_val, 4), round(profit, 4)],
                 fitness=round(f_val, 5),
                 served=len(cs['served']))

        avg_fit = sum(s["fitness"] for s in per_scen) / len(per_scen)
        all_results[inst_name] = {
            "best_fitness_search": round(best_fit, 5),
            "avg_final_fitness":   round(avg_fit,  5),
            "evaluations":         aco.evaluation_count,
            "elapsed_s":           round(elapsed, 1),
            "per_scenario":        per_scen,
            "fitness_history":     aco.history,
        }
        print(f"\n  → avg_fitness={avg_fit:.5f}  "
              f"evals={aco.evaluation_count}  time={elapsed:.1f}s")

        # Emit instance summary log
        _log("ACO", "instance_summary",
             avg_final_fitness=round(avg_fit, 5),
             best_fitness_search=round(best_fit, 5),
             evaluations=aco.evaluation_count,
             elapsed_s=round(elapsed, 1))
        _log("MAIN", "instance_done", instance=inst_name)

    print(f"\n{'='*65}")
    print(" SUMMARY")
    print(f"{'='*65}")
    print(f" {'Instance':<22} {'AvgFitness':>12} {'Evals':>8} {'Time(s)':>8}")
    print(f" {'-'*22} {'-'*12} {'-'*8} {'-'*8}")
    for name, res in all_results.items():
        print(f" {name:<22} {res['avg_final_fitness']:>12.5f} "
              f"{res['evaluations']:>8d} {res['elapsed_s']:>8.1f}")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(target)), "aco_results.json"
    )
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved → {out_path}")
    _log("MAIN", "all_done", instances_run=len(all_results))


if __name__ == "__main__":
    main()
