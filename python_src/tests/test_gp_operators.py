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
from sim.ctx import RoutingContext, SequencingContext


class GPOperatorTests(unittest.TestCase):
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
