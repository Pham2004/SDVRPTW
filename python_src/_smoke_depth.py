import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from gp import mod as gp_mod
from sim import ctx as sim_ctx

def tree_depth(gpc, p):
    return gpc.depth_to_bottom(p, 0)

def run(rd, sd, pop_size=60, gens=12, seed=123):
    rng = random.Random(seed)
    gpc_r = gp_mod.GPContext(rng=rng, num_population=pop_size, max_depth=rd)
    gpc_s = gp_mod.GPContext(rng=rng, num_population=pop_size, max_depth=sd)

    r_ctx = sim_ctx.RoutingContext
    s_ctx = sim_ctx.SequencingContext

    r_pop = gpc_r.ramp_half_and_half(context=r_ctx)
    s_pop = gpc_s.ramp_half_and_half(context=s_ctx)
    assert len(r_pop) == pop_size, f"routing pop {len(r_pop)} != {pop_size}"
    assert len(s_pop) == pop_size, f"seq pop {len(s_pop)} != {pop_size}"

    # check initial depth bounds
    for p in r_pop:
        d = tree_depth(gpc_r, p)
        assert d <= rd, f"init routing depth {d} > {rd}"
    for p in s_pop:
        d = tree_depth(gpc_s, p)
        assert d <= sd, f"init seq depth {d} > {sd}"

    # stress crossover + mutation many times
    max_seen_r = max(tree_depth(gpc_r, p) for p in r_pop)
    max_seen_s = max(tree_depth(gpc_s, p) for p in s_pop)
    for g in range(gens):
        for _ in range(pop_size):
            a, b = rng.randrange(pop_size), rng.randrange(pop_size)
            r1, r2 = gpc_r.crossover(r_pop[a], r_pop[b])
            s1, s2 = gpc_s.crossover(s_pop[a], s_pop[b])
            for d in (tree_depth(gpc_r, r1), tree_depth(gpc_r, r2)):
                assert d <= rd, f"xover routing depth {d} > {rd}"
                max_seen_r = max(max_seen_r, d)
            for d in (tree_depth(gpc_s, s1), tree_depth(gpc_s, s2)):
                assert d <= sd, f"xover seq depth {d} > {sd}"
                max_seen_s = max(max_seen_s, d)
            mr = gpc_r.mutation(r_pop[a])
            ms = gpc_s.mutation(s_pop[a])
            dr, ds = tree_depth(gpc_r, mr), tree_depth(gpc_s, ms)
            assert dr <= rd, f"mut routing depth {dr} > {rd}"
            assert ds <= sd, f"mut seq depth {ds} > {sd}"
            max_seen_r = max(max_seen_r, dr)
            max_seen_s = max(max_seen_s, ds)
            r_pop[a], s_pop[a] = r1, s1
    print(f"  rd={rd:2d} sd={sd:2d} -> OK | max routing depth seen={max_seen_r}, max seq depth seen={max_seen_s}", flush=True)

print("Stress-testing asymmetric max_depth combos:", flush=True)
for rd, sd in [(4,13),(13,4),(2,10),(10,2),(1,6),(6,1),(3,8)]:
    run(rd, sd)
print("ALL OK", flush=True)
