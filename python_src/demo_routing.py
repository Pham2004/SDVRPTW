#!/usr/bin/env python3
"""Demo: định tuyến ĐỘNG với best_tree tiến hóa từ GP.

Ý tưởng demo (chỉ Màn "routing động"):
  1. Chạy một vòng GP gọn (có seed) trên vài scenario để tiến hóa ra
     cặp quy tắc tốt nhất  (routing tree + sequencing tree) = `best_tree`.
  2. Dùng đúng best_tree đó chạy `Simulation` trên MỘT scenario mục tiêu.
  3. Vẽ animation:
       - customer hiện DẦN theo thời điểm request được biết (tính động),
       - mỗi xe chạy dọc tuyến do policy quyết định,
       - đổi màu điểm đã-phục-vụ, đếm số request đã lộ / đã phục vụ / quãng đường.

Cách chạy:
    python demo_routing.py                         # dùng scenario mặc định
    python demo_routing.py <scenario.csv>
    python demo_routing.py <scenario.csv> --gens 15 --pop 50 --seed 42
    python demo_routing.py <scenario.csv> --out demo_out/route.gif

Xuất file GIF (mặc định) hoặc MP4 nếu có ffmpeg và đuôi .mp4.
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import math
import random
import argparse

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Console Windows mặc định cp1252 -> ép UTF-8 để in được tiếng Việt.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render ra file, không cần màn hình
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from gp import mod as gp_mod
from sim import mod as sim_mod
from sim import ctx as sim_ctx
from sim.problem import Problem, ProblemSet

# Tái dùng cấu hình + Individual + fitness từ main để best_tree khớp pipeline thật.
import main as main_mod
from main import Individual, fitness, NUM_TIME_SLOT


# Nhãn dễ đọc cho terminal (chỉ phục vụ việc in best_tree cho người xem).
_ROUTING_TERMS = ["qLen", "capLeft", "distMedQ", "timeCost", "demand"]
_SEQ_TERMS = ["timeCost", "waitInQueue", "slack", "demand", "waitOpen",
              "revealTime", "-profit"]
sim_ctx.RoutingContext.format_terminal = staticmethod(lambda i: _ROUTING_TERMS[i])
sim_ctx.SequencingContext.format_terminal = staticmethod(lambda i: _SEQ_TERMS[i])


# ── Chọn cấu hình đội xe theo số customer (giống main.py) ────────────────────
_DATASET_CFG = {100: (10, 200), 200: (50, 400), 400: (100, 800)}


def _auto_fleet(csv_path: str):
    with open(csv_path, newline='') as fh:
        n_rows = sum(1 for r in csv.reader(fh) if r and any(c.strip() for c in r))
    n_customers = max(n_rows - 2, 0)  # trừ header + depot
    size = min(_DATASET_CFG, key=lambda s: abs(s - n_customers))
    trucks, cap = _DATASET_CFG[size]
    return n_customers, trucks, cap


# ── GP gọn: tiến hóa ra best_tree ────────────────────────────────────────────
def _select_parent(rng, pop):
    idxs = rng.sample(range(len(pop)), k=min(8, len(pop)))
    return min(idxs, key=lambda i: pop[i].result[2])


def evolve_best_policy(train_set, gens, pop_size, max_depth, seed):
    """Trả về (best_individual, lịch_sử_fitness_tốt_nhất)."""
    rng = random.Random(seed)
    gpc_r = gp_mod.GPContext(rng=rng, num_population=pop_size, max_depth=max_depth)
    gpc_s = gp_mod.GPContext(rng=rng, num_population=pop_size, max_depth=max_depth)
    train_ps = train_set if isinstance(train_set, ProblemSet) else ProblemSet(list(train_set))
    time_slots = [p.depot.close / NUM_TIME_SLOT for p in train_ps]
    cache: dict = {}
    pop = Individual.ramp_half_and_half(gpc_r, gpc_s)

    history = []
    for gen in range(1, gens + 1):
        for ind in pop:
            ind.evaluate(cache, train_ps, time_slots)
        pop.sort(key=lambda i: i.result[2])       # fitness nhỏ = tốt hơn
        pop = pop[:pop_size]
        history.append(pop[0].result[2])
        print(f"  gen {gen:>3}/{gens}  best_fitness={pop[0].result[2]:.5f}")

        new_pop = list(pop)
        for _ in range(pop_size // 2):
            p1 = _select_parent(rng, pop)
            p2 = _select_parent(rng, pop)
            x = rng.random()
            if x <= main_mod.CROSSOVER_RATE:
                c1, c2 = pop[p1].crossover_with(gpc_r, gpc_s, pop[p2])
                new_pop.extend([c1, c2])
            elif x <= main_mod.CROSSOVER_RATE + main_mod.MUTATION_RATE:
                new_pop.extend([pop[p1].mutate(gpc_r, gpc_s), pop[p2].mutate(gpc_r, gpc_s)])
            else:
                new_pop.extend([pop[p1], pop[p2]])
        pop = new_pop

    for ind in pop:
        ind.evaluate(cache, train_ps, time_slots)
    pop.sort(key=lambda i: i.result[2])
    return pop[0], history


# ── Trích dữ liệu animation từ một lần mô phỏng ───────────────────────────────
def build_frames(problem, best, time_slot):
    """Chạy Simulation với best_tree, trả về mọi thứ cần để vẽ."""
    sim = sim_mod.Simulation(problem, best.routing, best.sequencing)
    dist, profit = sim.simulate_until(time_slot, float("inf"))

    # Vị trí các node theo idx (0 = depot)
    pos = {0: (problem.depot.x, problem.depot.y)}
    reveal = {}          # idx -> thời điểm request được biết
    for r in problem.requests:
        pos[r.idx] = (r.x, r.y)
        reveal[r.idx] = r.time

    # Mỗi xe: chuỗi (thời điểm đến, idx). Bắt đầu tại depot lúc t=0.
    vehicles = []
    served_at = {}       # idx (khác depot) -> thời điểm được phục vụ sớm nhất
    horizon = 0.0
    for v in sim.vehicles:
        stops = [(0.0, 0)] + sorted(v.route.items(), key=lambda kv: kv[0])
        ts = np.array([t for t, _ in stops], dtype=float)
        xy = np.array([pos.get(i, pos[0]) for _, i in stops], dtype=float)
        vehicles.append({"ts": ts, "xy": xy, "idx": [i for _, i in stops]})
        horizon = max(horizon, ts[-1] if len(ts) else 0.0)
        for t, i in stops:
            if i != 0:
                served_at[i] = min(served_at.get(i, math.inf), t)

    horizon = max(horizon, problem.depot.close)
    return {
        "pos": pos, "reveal": reveal, "vehicles": vehicles,
        "served_at": served_at, "horizon": horizon,
        "distance": dist, "profit": profit,
        "n_requests": len(problem.requests),
    }


def print_arrival_times(data):
    """In thời gian đến từng điểm của mỗi xe."""
    print("\n=== THỜI GIAN ĐẾN CỦA CÁC XE ===")
    for k, veh in enumerate(data["vehicles"], start=1):
        # bỏ mốc xuất phát tại depot lúc t=0
        stops = [(i, t) for t, i in zip(veh["ts"], veh["idx"])][1:]
        if not stops:
            print(f"vehicle {k}: (không rời depot)")
            continue
        body = ", ".join(
            f"{'depot' if i == 0 else i} ({t:.2f})" for i, t in stops
        )
        print(f"vehicle {k}: {body}")


def _veh_state(veh, t):
    """Vị trí xe tại thời điểm t + số stop đã đi qua (để vẽ vệt đường)."""
    ts, xy = veh["ts"], veh["xy"]
    if t <= ts[0]:
        return xy[0], 0
    if t >= ts[-1]:
        return xy[-1], len(ts) - 1
    j = int(np.searchsorted(ts, t, side="right")) - 1
    j = max(0, min(j, len(ts) - 2))
    span = ts[j + 1] - ts[j]
    frac = 0.0 if span <= 0 else (t - ts[j]) / span
    p = xy[j] + frac * (xy[j + 1] - xy[j])
    return p, j


# ── Vẽ animation ─────────────────────────────────────────────────────────────
def animate(data, out_path, n_frames=160, fps=20, title=""):
    pos = data["pos"]; reveal = data["reveal"]
    vehicles = data["vehicles"]; served_at = data["served_at"]
    horizon = data["horizon"]

    req_ids = [i for i in pos if i != 0]
    rx = np.array([pos[i][0] for i in req_ids])
    ry = np.array([pos[i][1] for i in req_ids])
    reveal_t = np.array([reveal[i] for i in req_ids])
    served_t = np.array([served_at.get(i, math.inf) for i in req_ids])

    all_x = np.array([p[0] for p in pos.values()])
    all_y = np.array([p[1] for p in pos.values()])
    pad = 0.06 * (all_x.max() - all_x.min() + 1e-9)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, color="w")

    C_HIDDEN, C_PENDING, C_SERVED = "#33373f", "#ffd24a", "#2ecc71"
    cmap = plt.get_cmap("tab10")

    # Depot
    dx, dy = pos[0]
    ax.scatter([dx], [dy], marker="s", s=220, c="#e74c3c",
               edgecolors="w", linewidths=1.5, zorder=6)
    ax.annotate("DEPOT", (dx, dy), textcoords="offset points",
                xytext=(0, 12), ha="center", color="w", fontsize=9)

    scat = ax.scatter(rx, ry, s=40, c=[C_HIDDEN] * len(req_ids),
                      edgecolors="none", zorder=4)

    trails, markers = [], []
    for k in range(len(vehicles)):
        col = cmap(k % 10)
        (ln,) = ax.plot([], [], "-", color=col, lw=1.6, alpha=0.85, zorder=3)
        (mk,) = ax.plot([], [], marker=">", color=col, ms=11,
                        mec="w", mew=1.0, zorder=7, ls="")
        trails.append(ln); markers.append(mk)

    hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=11, color="w", family="monospace",
                  bbox=dict(boxstyle="round", fc="#111417", ec="#444", alpha=0.85))

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#e74c3c",
               markeredgecolor="w", markersize=10, label="Depot"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_HIDDEN,
               markersize=9, label="Chưa lộ"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PENDING,
               markersize=9, label="Chờ phục vụ"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_SERVED,
               markersize=9, label="Đã phục vụ"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)

    def dist_up_to(veh, t):
        ts, xy = veh["ts"], veh["xy"]
        pcur, j = _veh_state(veh, t)
        d = 0.0
        for a in range(j):
            d += float(np.hypot(*(xy[a + 1] - xy[a])))
        d += float(np.hypot(*(pcur - xy[j])))
        return d

    def update(f):
        t = horizon * f / max(1, n_frames - 1)

        colors = np.where(reveal_t > t, C_HIDDEN,
                          np.where(served_t <= t, C_SERVED, C_PENDING))
        scat.set_color(colors)

        total_d = 0.0
        for veh, ln, mk in zip(vehicles, trails, markers):
            p, j = _veh_state(veh, t)
            xs = list(veh["xy"][:j + 1, 0]) + [p[0]]
            ys = list(veh["xy"][:j + 1, 1]) + [p[1]]
            ln.set_data(xs, ys)
            mk.set_data([p[0]], [p[1]])
            total_d += dist_up_to(veh, t)

        revealed = int((reveal_t <= t).sum())
        served = int((served_t <= t).sum())
        hud.set_text(
            f"t = {t:7.1f} / {horizon:.0f}\n"
            f"đã lộ    : {revealed:3d} / {len(req_ids)}\n"
            f"đã phục vụ: {served:3d} / {len(req_ids)}\n"
            f"quãng đường: {total_d:8.1f}"
        )
        return [scat, hud, *trails, *markers]

    print(f"  render {n_frames} frame ...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if out_path.lower().endswith(".mp4"):
        try:
            anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        except Exception as e:
            out_path = out_path[:-4] + ".gif"
            print(f"  (không có ffmpeg: {e}) -> lưu GIF thay thế")
            anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────
def _default_scenario():
    hits = sorted(glob.glob(os.path.join(REPO_ROOT, "datasets", "**", "*.csv"),
                            recursive=True))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Demo định tuyến động với best_tree (GP).")
    ap.add_argument("scenario", nargs="?", default=None, help="đường dẫn scenario CSV")
    ap.add_argument("--gens", type=int, default=12, help="số thế hệ GP (mặc định 12)")
    ap.add_argument("--pop", type=int, default=40, help="kích thước quần thể (mặc định 40)")
    ap.add_argument("--depth", type=int, default=6, help="độ sâu cây tối đa")
    ap.add_argument("--seed", type=int, default=42, help="seed cho tái lập")
    ap.add_argument("--train", type=int, default=3, help="số scenario cùng thư mục dùng train")
    ap.add_argument("--frames", type=int, default=160, help="số frame animation")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "demo_out", "route_demo.gif"))
    ap.add_argument("--trucks", type=int, default=None)
    ap.add_argument("--cap", type=float, default=None)
    args = ap.parse_args()

    scenario = args.scenario or _default_scenario()
    if not scenario or not os.path.isfile(scenario):
        print(f"[ERROR] không tìm thấy scenario CSV: {scenario}")
        sys.exit(1)

    n_cust, trucks, cap = _auto_fleet(scenario)
    if args.trucks: trucks = args.trucks
    if args.cap:    cap = args.cap
    print(f"Scenario   : {scenario}")
    print(f"Customers  : {n_cust}  ->  num_trucks={trucks}, capacity={cap}")

    # Tập train: scenario mục tiêu + vài scenario cùng thư mục.
    folder = os.path.dirname(scenario)
    siblings = sorted(glob.glob(os.path.join(folder, "*.csv")))
    train_paths = [scenario] + [p for p in siblings if p != scenario][:max(0, args.train - 1)]
    train_set = [Problem.load(p, 1.0, float(cap), trucks) for p in train_paths]
    target = train_set[0]

    print(f"Train trên : {len(train_set)} scenario, GP pop={args.pop} gens={args.gens} "
          f"seed={args.seed}")
    print("Tiến hóa best_tree ...")
    best, history = evolve_best_policy(train_set, args.gens, args.pop, args.depth, args.seed)

    # In ra best_tree (biểu thức có tên terminal/operator thật)
    r_expr = best.routing.fmt(sim_ctx.RoutingContext)
    s_expr = best.sequencing.fmt(sim_ctx.SequencingContext)
    print("\n=== BEST TREE ===")
    print(f"  fitness (train): {best.result[2]:.5f}  "
          f"(dist={best.result[0]:.1f}, profit={best.result[1]:.1f})")
    print(f"  routing   : {r_expr}")
    print(f"  sequencing: {s_expr}")

    time_slot = target.depot.close / NUM_TIME_SLOT
    data = build_frames(target, best, time_slot)
    print(f"\nMô phỏng scenario mục tiêu: dist={data['distance']:.1f} "
          f"profit={data['profit']:.1f}  horizon={data['horizon']:.0f}")

    inst = os.path.splitext(os.path.basename(scenario))[0]
    title = (f"SDVRPTW – định tuyến động (best_tree GP)  |  {inst}  |  "
             f"{trucks} xe, {n_cust} khách")
    out = animate(data, args.out, n_frames=args.frames, fps=args.fps, title=title)
    print(f"\n✓ Đã lưu animation: {out}")


if __name__ == "__main__":
    main()
