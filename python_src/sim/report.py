"""In lộ trình và thống kê của một lời giải đã mô phỏng.

`Simulation.get_routes()` trả về, cho mỗi xe, dãy `request.idx` theo thứ tự
phục vụ (0 = depot, xuất hiện giữa route khi xe quay về nạp hàng). Module này
gộp dãy đó với `problem` để quy ra demand/profit từng xe.
"""
from __future__ import annotations

from typing import List


def print_solution(problem, routes: List[List[int]], total_distance: float,
                   total_profit: float, title: str = "") -> None:
    req_map = {r.idx: r for r in problem.requests}

    cap = problem.truck_capacity

    if title:
        print(f"\n=== {title} ===")
    print(f"{'Xe':>3}  {'Lộ trình':<40} {'Demand':>14} {'Profit':>8}")
    print("-" * 70)

    served = set()
    sum_demand = 0.0
    idle = 0
    for v, route in enumerate(routes):
        if not route:
            idle += 1
            continue

        demand = sum(req_map[i].demand for i in route if i in req_map)
        profit = sum(req_map[i].profit for i in route if i in req_map)
        served.update(i for i in route if i in req_map)
        sum_demand += demand

        path = " -> ".join(["0"] + [str(i) for i in route] + ["0"])
        if len(path) > 40:
            path = path[:37] + "..."
        # Xe về depot giữa route (idx 0) sẽ nạp lại hàng, nên demand tích lũy
        # của cả lộ trình có thể vượt capacity một cách hợp lệ.
        load = f"{demand:>6.1f}/{cap:<6.0f}"
        print(f"{v:>3}  {path:<40} {load:>14} {profit:>8.2f}")

    print("-" * 70)
    if idle:
        print(f"({idle} xe không xuất phát)")
    print(f"Tổng demand đã giao : {sum_demand:.1f} / {problem.total_demand():.1f}")
    print(f"Tổng profit         : {total_profit:.2f}")
    print(f"Tổng quãng đường    : {total_distance:.2f}")

    unserved = sorted(set(req_map) - served)
    if unserved:
        print(f"Đỉnh không phục vụ  : {unserved}")


__all__ = ["print_solution"]
