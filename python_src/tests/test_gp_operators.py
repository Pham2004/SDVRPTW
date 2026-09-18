import os
import random
import sys
import unittest

PYTHON_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_SRC)
for path in (PYTHON_SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from gp.mod import GPContext
from main import select_parent
from run_gp_pipeline import _resolved_depths
from sim.ctx import RoutingContext, SequencingContext


class GPOperatorTests(unittest.TestCase):
    def test_pipeline_defaults_to_distinct_tree_depths(self):
        class Args:
            max_depth = None
            routing_depth = None
            sequencing_depth = None

        self.assertEqual(_resolved_depths(Args()), (8, 6))

    def test_tournament_selection_minimizes_fitness(self):
        class Individual:
            def __init__(self, fitness):
                self.result = (0.0, 0.0, fitness)

        gpc = GPContext(random.Random(123), num_population=8, max_depth=2)
        population = [Individual(float(i)) for i in range(8)]
        self.assertEqual(select_parent(gpc, population), 0)

    def test_generation_crossover_and_mutation_preserve_tree_invariants(self):
        for context in (RoutingContext, SequencingContext):
            gpc = GPContext(random.Random(123), num_population=20, max_depth=6)
            population = gpc.ramp_half_and_half(context=context)
            self.assertEqual(len(population), 20)
            for program in population:
                program.verify()
                self.assertLessEqual(gpc.depth_to_bottom(program, 0), 6)

            for i in range(50):
                p1 = population[i % len(population)]
                p2 = population[(i * 7 + 1) % len(population)]
                child1, child2 = gpc.crossover(p1, p2)
                mutated = gpc.mutation(p1)
                for program in (child1, child2, mutated):
                    program.verify()
                    self.assertLessEqual(gpc.depth_to_bottom(program, 0), 6)


if __name__ == "__main__":
    unittest.main()
