import os
import sys
import unittest

PYTHON_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_SRC)
for path in (PYTHON_SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from gp.GPtree import Program
from sim.mod import Simulation
from sim.problem import Problem, Request


def request(idx, x, y, demand=1.0, open_time=0.0, close=100.0,
            service=0.0, reveal=0.0, profit=1.0):
    return Request(idx, x, y, demand, open_time, close, service,
                   reveal, profit, 0)


def problem(requests, capacity=10.0, speed=1.0, trucks=1):
    depot = request(0, 0.0, 0.0, demand=0.0, close=100.0, profit=0.0)
    return Problem(depot, requests, speed, capacity, trucks)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        # raw travel time for routing, raw travel time for sequencing
        self.routing = Program.terminal(3)
        self.sequencing = Program.terminal(0)

    def test_distance_profit_and_route_are_hand_checkable(self):
        p = problem([request(1, 3.0, 4.0, profit=10.0)])
        sim = Simulation(p, self.routing, self.sequencing)
        distance, profit = sim.simulate_until(5.0, float("inf"))
        self.assertAlmostEqual(distance, 10.0)
        self.assertAlmostEqual(profit, 10.0)
        self.assertEqual(sim.get_routes(), [[1]])
        self.assertEqual(sim.dropped_requests, {})

    def test_request_is_revealed_at_next_slot(self):
        p = problem([request(1, 1.0, 0.0, reveal=1.1, close=10.0)])
        sim = Simulation(p, self.routing, self.sequencing)
        distance, profit = sim.simulate_until(5.0, float("inf"))
        # The request is dispatched at slot 5 and completed at 6.  The final
        # depot leg is accounted in distance but is not another processed event.
        self.assertEqual(sim.time, 6.0)
        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(profit, 1.0)

    def test_oversize_request_is_dropped_without_zero_time_loop(self):
        p = problem([request(1, 0.0, 0.0, demand=11.0)], capacity=10.0)
        sim = Simulation(p, self.routing, self.sequencing)
        distance, profit = sim.simulate_until(5.0, float("inf"))
        self.assertEqual((distance, profit), (0.0, 0.0))
        self.assertEqual(
            sim.dropped_requests[1], "demand_exceeds_vehicle_capacity"
        )

    def test_equal_timestamps_do_not_overwrite_visits(self):
        p = problem([
            request(1, 0.0, 0.0),
            request(2, 0.0, 0.0),
        ])
        sim = Simulation(p, self.routing, self.sequencing)
        _distance, profit = sim.simulate_until(5.0, float("inf"))
        self.assertEqual(profit, 2.0)
        self.assertEqual(sim.get_routes(), [[1, 2]])

    def test_simulation_is_explicitly_single_use(self):
        sim = Simulation(problem([]), self.routing, self.sequencing)
        sim.simulate_until(5.0, float("inf"))
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            sim.simulate_until(5.0, float("inf"))

    def test_invalid_vehicle_configuration_fails_fast(self):
        sim = Simulation(problem([], speed=0.0), self.routing, self.sequencing)
        with self.assertRaisesRegex(ValueError, "truck_speed"):
            sim.simulate_until(5.0, float("inf"))


if __name__ == "__main__":
    unittest.main()
