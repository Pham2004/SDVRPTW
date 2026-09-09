"""Policy adapter that runs PPO decisions through the canonical simulator.

The GP code makes two kinds of decisions: assign each revealed request to a
vehicle (routing), then select the next request from a vehicle queue
(sequencing).  ``PolicySimulation`` keeps exactly those decision points and
only replaces the two GP tree scorers by a masked neural policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    from ..sim.mod import Simulation, VehicleState
except ImportError:
    from sim.mod import Simulation, VehicleState


FEATURE_NAMES = (
    "is_sequencing",
    "queue_fraction",
    "remaining_capacity",
    "busy_delay",
    "vehicle_x",
    "vehicle_y",
    "delta_x",
    "delta_y",
    "distance",
    "travel_time",
    "request_demand",
    "window_open",
    "window_close",
    "service_time",
    "reveal_time",
    "request_profit",
    "arrival_slack",
    "request_wait",
    "current_time",
)


@dataclass
class TrajectoryStep:
    decision_type: str
    features: List[List[float]]
    action: int
    log_prob: float
    value: float
    reward: float = 0.0


class PolicySimulation(Simulation):
    """A ``Simulation`` whose two dispatch rules are selected by a policy.

    The shaped reward is algebraically equivalent to the repository fitness:

      sum(reward) = -w * distance/max_distance
                    + (1-w) * profit/max_profit

    Therefore maximizing return is exactly the same ordering as minimizing
    ``main.fitness`` (the omitted ``1-w`` term is constant).
    """

    def __init__(self, problem, policy, weight: float = 0.5,
                 deterministic: bool = False):
        super().__init__(problem, routing_rule=None, sequencing_rule=None)
        self.policy = policy
        self.weight = float(weight)
        self.deterministic = deterministic
        self.steps: List[TrajectoryStep] = []
        self._reward_anchor: Optional[int] = None

        nodes = [problem.depot] + list(problem.requests)
        self._horizon = max(
            [1.0, float(problem.depot.close)]
            + [float(r.close) for r in problem.requests]
        )
        xs = [float(r.x) for r in nodes]
        ys = [float(r.y) for r in nodes]
        self._coord_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        self._max_profit = max(sum(max(float(r.profit), 0.0)
                                   for r in problem.requests), 1.0)
        self._distance_scale = max(
            problem.truck_speed * problem.depot.close * problem.num_trucks,
            1.0,
        )

    def _features(self, state: VehicleState, request, time: float,
                  is_sequencing: bool) -> List[float]:
        p = self.problem
        distance = state.distance_to(request)
        travel_time = distance / p.truck_speed
        depart = max(time, state.busy_until)
        arrival = depart + travel_time
        cap = max(float(p.truck_capacity), 1.0)
        nreq = max(len(p.requests), 1)
        h = self._horizon
        s = self._coord_scale
        dx = float(request.x) - float(state.cur_request.x)
        dy = float(request.y) - float(state.cur_request.y)
        return [
            1.0 if is_sequencing else 0.0,
            len(state.queue) / nreq,
            float(state.total_demand) / cap,
            max(float(state.busy_until) - time, 0.0) / h,
            (float(state.cur_request.x) - float(p.depot.x)) / s,
            (float(state.cur_request.y) - float(p.depot.y)) / s,
            dx / s,
            dy / s,
            distance / s,
            travel_time / h,
            float(request.demand) / cap,
            float(request.open) / h,
            float(request.close) / h,
            float(request.service_time) / h,
            float(request.time) / h,
            float(request.profit) / self._max_profit,
            (float(request.close) - arrival) / h,
            max(time - float(request.open), 0.0) / h,
            time / h,
        ]

    def _choose(self, decision_type: str,
                features: Sequence[Sequence[float]]) -> int:
        action, log_prob, value = self.policy.select_action(
            features, deterministic=self.deterministic
        )
        self.steps.append(TrajectoryStep(
            decision_type=decision_type,
            features=[list(row) for row in features],
            action=int(action),
            log_prob=float(log_prob),
            value=float(value),
        ))
        self._reward_anchor = len(self.steps) - 1
        return int(action)

    def _add_reward(self, distance: float, profit: float) -> None:
        if self._reward_anchor is None:
            return
        reward = (
            -self.weight * distance / self._distance_scale
            + (1.0 - self.weight) * profit / self._max_profit
        )
        self.steps[self._reward_anchor].reward += reward

    def routing_rule_route_request(self, problem, time: float,
                                   vehicles: List[VehicleState], request):
        candidate_vehicle_ids = []
        candidate_features = []
        for vehicle_id, state in enumerate(vehicles):
            earliest_departure = max(time, state.busy_until)
            travel_time = state.distance_to(request) / problem.truck_speed
            if earliest_departure + travel_time <= request.close:
                candidate_vehicle_ids.append(vehicle_id)
                candidate_features.append(
                    self._features(state, request, time, is_sequencing=False)
                )
        if not candidate_vehicle_ids:
            return None
        local_action = self._choose("routing", candidate_features)
        return candidate_vehicle_ids[local_action]

    def sequencing_rule_sequence_request(self, problem, time: float,
                                          vehicle_state: VehicleState, cache):
        if not vehicle_state.queue:
            return None
        features = [
            self._features(vehicle_state, request, time, is_sequencing=True)
            for request, _ready_time in vehicle_state.queue
        ]
        return self._choose("sequencing", features)

    def route_vehicle_to(self, vehicle: int, request, _cb,
                         total_distance_container,
                         total_profit_container=None) -> None:
        state = self.vehicles[vehicle]
        distance = state.distance_to(request)
        profit = 0.0 if getattr(request, "idx", 0) == 0 else float(
            getattr(request, "profit", 0.0)
        )
        super().route_vehicle_to(
            vehicle, request, _cb, total_distance_container,
            total_profit_container,
        )
        self._add_reward(distance, profit)

    def routing_rule_route_request(self, problem, time: float,
                                   vehicles: List[VehicleState], request):
        candidates = []
        feature_rows = []
        for vehicle_idx, state in enumerate(vehicles):
            earliest_departure = max(time, state.busy_until)
            travel = state.raw_time_cost(problem, request, time)
            if earliest_departure + travel <= request.close:
                candidates.append(vehicle_idx)
                feature_rows.append(self._features(state, request, time, False))
        if not candidates:
            return None
        return candidates[self._choose("routing", feature_rows)]

    def sequencing_rule_sequence_request(self, problem, time: float,
                                          vehicle_state: VehicleState,
                                          cache):
        del cache
        if not vehicle_state.queue:
            return None
        feature_rows = [
            self._features(vehicle_state, request, time, True)
            for request, _ready_time in vehicle_state.queue
        ]
        return self._choose("sequencing", feature_rows)

    def route_vehicle_to(self, vehicle: int, request, _cb,
                         total_distance_container,
                         total_profit_container=None) -> None:
        state = self.vehicles[vehicle]
        distance = state.distance_to(request)
        profit = 0.0 if request.idx == 0 else float(request.profit)
        super().route_vehicle_to(
            vehicle, request, _cb, total_distance_container,
            total_profit_container,
        )
        self._add_reward(distance, profit)


__all__ = ["FEATURE_NAMES", "PolicySimulation", "TrajectoryStep"]
