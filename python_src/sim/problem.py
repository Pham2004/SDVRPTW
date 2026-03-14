"""
Problem representation and CSV loader (Python port of src/sim/problem.rs).
Responsibilities:
- `Request`: simple data holder for a single request (position, demand,
    time windows and service properties).
- `Problem`: holds `depot`, `requests`, and vehicle parameters and provides
    CSV loading via `Problem.load(...)` and training-time cloning via
    `clone_training(...)` used to build smaller/stressed training problems.

Notes:
- `Problem.load` expects the dataset CSV format used by the project and
    extracts fields into `Request` instances; the first row is treated as
    the depot.
"""
from __future__ import annotations

from typing import List
import csv
import torch
from problems._data_dtw import DVRPTW_Dataset 
                

# mỗi request là một dòng trong file CSV
class Request:
    def __init__(self, idx: int, x: float, y: float, demand: float, open: float, close: float, service_time: float, time: float, profit: float, type: int):
        self.idx = (idx)
        self.x = (x)
        self.y = (y)
        self.demand = (demand)
        self.open = (open)
        self.close = (close)
        self.service_time = (service_time)
        self.time = (time)
        self.profit = (profit)
        self.type = (type)


class Problem:
    def __init__(self, depot: Request, requests: List[Request], truck_speed: float, truck_capacity: float, num_trucks: int):
        self.depot = depot
        self.requests = requests
        self.truck_speed = (truck_speed)
        self.truck_capacity = (truck_capacity)
        self.num_trucks = (num_trucks)

    @staticmethod
    def load(csv_path: str, truck_speed: float, truck_capacity: float, num_trucks: int) -> "Problem":
        requests: List[Request] = []
        with open(csv_path, newline='') as fh:
            reader = csv.reader(fh)
            # skip header
            try:
                next(reader)
            except StopIteration:
                raise ValueError("CSV is empty")
            for idx, row in enumerate(reader):
                if not row or all(not c.strip() for c in row):
                    continue
                # parse floats, allow extra columns
                vals = [float(tok) for tok in row]
                req = Request(
                    idx=idx,
                    x=float(vals[0]),
                    y=float(vals[1]),
                    demand=float(vals[2]),
                    open=float(vals[3]),
                    close=float(vals[4]),
                    service_time= float(vals[5]),
                    time=float(vals[7]),
                    profit=float(vals[8]) if len(vals) > 8 else 0.0,
                    type=int(float(vals[9])) if len(vals) > 9 else 0,
                )
                requests.append(req)

        if not requests:
            raise ValueError("no requests found in CSV")
        depot = requests.pop(0)
        return Problem(depot, requests, truck_speed, truck_capacity, num_trucks)
    
    @staticmethod
    def tensor_load(tensor_load: str, instance_num: int = 0, truck_speed: float = 1.0, truck_capacity: float = 1.0, num_trucks: int = 1) -> "Problem":
        # Load a dataset saved by the tools/data pipeline (torch file)
        torch.serialization.add_safe_globals([DVRPTW_Dataset])
        ds = torch.load(tensor_load, weights_only=False)

        # Extract nodes tensor (support both dict and object with attribute)
        if isinstance(ds, dict) and 'nodes' in ds:
            nodes = ds['nodes']
        elif hasattr(ds, 'nodes'):
            nodes = ds.nodes
        else:
            raise ValueError("dataset does not contain 'nodes' tensor")

        # If nodes has a batch dimension, select the requested scenario
        if nodes.dim() == 3:
            batch_size = nodes.size(0)
            if not (0 <= instance_num < batch_size):
                raise IndexError(f"instance_num {instance_num} out of range (batch_size={batch_size})")
            inst = nodes[instance_num]
        else:
            # single-instance tensor, ignore instance_num
            inst = nodes

        n_nodes, feat = inst.size()

        # Read depot (first row)
        depot_row = inst[0]
        depot_x = float(depot_row[0].item()) if feat >= 1 else 0.0
        depot_y = float(depot_row[1].item()) if feat >= 2 else 0.0
        depot_close = float(depot_row[4].item()) if feat >= 5 else float(0.0)
        depot = Request(idx=0, x=depot_x, y=depot_y, demand=0.0, open=0.0, close=depot_close, service_time=0.0, time=0.0)

        requests: List[Request] = []
        for j in range(1, n_nodes):
            row = inst[j]
            x = float(row[0].item()) if feat >= 1 else 0.0
            y = float(row[1].item()) if feat >= 2 else 0.0
            demand = float(row[2].item()) if feat >= 3 else 0.0
            open_t = float(row[3].item()) if feat >= 4 else 0.0
            close_t = float(row[4].item()) if feat >= 5 else open_t
            service_time = float(row[5].item()) if feat >= 6 else 0.0
            time = float(row[6].item()) if feat >= 7 else 0.0

            req = Request(idx=j, x=x, y=y, demand=demand, open=open_t, close=close_t, service_time=service_time, time=time)
            requests.append(req)

        #requests.sort(key=lambda x: x.time)
        return Problem(depot, requests, truck_speed, truck_capacity, num_trucks)

    def clone_training(self, time_limit: float, stress_factor: float) -> "Problem":
        requests: List[Request] = []
        current_index = 0
        turn = 0.0
        original_requests = list(self.requests)
        n = len(original_requests)
        for req in original_requests:
            nr = Request(
                idx=req.idx,
                x=req.x * stress_factor,
                y=req.y * stress_factor,
                demand=req.demand,
                open=req.open * stress_factor,
                close=req.close * stress_factor,
                service_time=req.service_time * stress_factor,
                time=req.time,
                profit=req.profit,
                type=req.type,
            )
            if nr.time > time_limit:
                # select a time reference request
                time_req = original_requests[current_index % n]
                nr.time = time_limit * turn + (time_req.time + time_req.open * 1.5) / 2.5
                nr.open = time_limit * turn + time_req.open
                nr.close = time_limit * turn + time_req.close

                current_index += 1
                if original_requests[current_index % n].time > time_limit:
                    current_index = 0
                    turn += 1.0

            requests.append(nr)

        return Problem(self.depot, requests, self.truck_speed, self.truck_capacity, self.num_trucks)

    def total_demand(self) -> float:
        return sum(r.demand for r in self.requests)


class ProblemSet:
    """
    Lớp chứa và quản lý nhiều instance (Problem) cùng lúc.
    Sử dụng để tính fitness trên nhiều instance, cho phép đánh giá tổng quan hơn.
    """
    def __init__(self, problems: List[Problem]):
        self.problems = problems
        self.results = []
        self.fitnesses = []

    @staticmethod
    def load_from_csvs(csv_paths: List[str], truck_speed: float, truck_capacity: float, num_trucks: int) -> "ProblemSet":
        """Khởi tạo một ProblemSet từ nhiều file CSV."""
        problems = [Problem.load(path, truck_speed, truck_capacity, num_trucks) for path in csv_paths]
        return ProblemSet(problems)

    def get_average_problem(self) -> Problem:
        """
        Create a single Problem that averages all attributes (x, y, demand, open, close, service_time)
        of requests across all problems in this set.
        Assumes all problems have the same number of requests and same truck configurations.
        """
        if not self.problems:
            raise ValueError("ProblemSet is empty")
        
        base_problem = self.problems[0]
        num_requests = len(base_problem.requests)
        num_problems = len(self.problems)

        avg_depot_x = sum(p.depot.x for p in self.problems) / num_problems
        avg_depot_y = sum(p.depot.y for p in self.problems) / num_problems
        avg_depot_demand = sum(p.depot.demand for p in self.problems) / num_problems
        avg_depot_open = sum(p.depot.open for p in self.problems) / num_problems
        avg_depot_close = sum(p.depot.close for p in self.problems) / num_problems
        avg_depot_service_time = sum(p.depot.service_time for p in self.problems) / num_problems
        avg_depot_time = sum(p.depot.time for p in self.problems) / num_problems
        avg_depot_profit = sum(getattr(p.depot, "profit", 0.0) for p in self.problems) / num_problems

        avg_depot = Request(
            idx=base_problem.depot.idx,
            x=avg_depot_x,
            y=avg_depot_y,
            demand=avg_depot_demand,
            open=avg_depot_open,
            close=avg_depot_close,
            service_time=avg_depot_service_time,
            time=avg_depot_time,
            profit=avg_depot_profit,
            type=base_problem.depot.type
        )

        avg_requests = []
        for i in range(num_requests):
            req_idx = base_problem.requests[i].idx
            req_type = base_problem.requests[i].type
            avg_x = sum(p.requests[i].x for p in self.problems) / num_problems
            avg_y = sum(p.requests[i].y for p in self.problems) / num_problems
            avg_demand = sum(p.requests[i].demand for p in self.problems) / num_problems
            avg_open = sum(p.requests[i].open for p in self.problems) / num_problems
            avg_close = sum(p.requests[i].close for p in self.problems) / num_problems
            avg_service_time = sum(p.requests[i].service_time for p in self.problems) / num_problems
            avg_time = sum(p.requests[i].time for p in self.problems) / num_problems
            avg_profit = sum(getattr(p.requests[i], "profit", 0.0) for p in self.problems) / num_problems

            avg_requests.append(Request(
                idx=req_idx,
                x=avg_x,
                y=avg_y,
                demand=avg_demand,
                open=avg_open,
                close=avg_close,
                service_time=avg_service_time,
                time=avg_time,
                profit=avg_profit,
                type=req_type
            ))

        return Problem(
            depot=avg_depot,
            requests=avg_requests,
            truck_speed=base_problem.truck_speed,
            truck_capacity=base_problem.truck_capacity,
            num_trucks=base_problem.num_trucks
        )

    def update_fitness(self, results: list, fitness_func) -> float:
        """
        Cập nhật khi định tuyến xong cho toàn bộ instance.
        - results: Danh sách các kết quả (ví dụ tuple (distance, profit)) cho từng instance
        - fitness_func: Hàm tính fitness (nhận vào problem, result và trả về float; ví dụ hàm fitness trong main.py)
        
        Tính toán fitness cho tất cả và trả về fitness trung bình của chúng.
        """
        if len(results) != len(self.problems):
            raise ValueError(f"Got {len(results)} results for {len(self.problems)} problems")
            
        self.results = results
        self.fitnesses = [fitness_func(p, r) for p, r in zip(self.problems, self.results)]
        return self.average_fitness()

    def average_fitness(self) -> float:
        if not self.fitnesses:
            return 0.0
        return sum(self.fitnesses) / len(self.fitnesses)

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Problem:
        return self.problems[idx]

    def __iter__(self):
        return iter(self.problems)


__all__ = ["Request", "Problem", "ProblemSet"]

