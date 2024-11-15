import libtspenvv2o3 as libtsp
from pathlib import Path
from pprint import pprint

blob = list(Path('.').rglob('tsp*.dat'))[0].read_text()
lines = list(blob.strip().split('\n'))

n_objs = int(lines[0].strip())
n_vars = int(lines[1].strip())
objs = []
_objs = []
count = 0
for line in lines[2:]:
    _objs.append(list(map(int, line.strip().split(" "))))
    count += 1
    
    if count == n_vars:
        objs.append(_objs)
        _objs = []
        count = 0

env = libtsp.TSPEnv()
env.reset(3)
env.set_inst(n_vars, n_objs, objs)
env.initialize_dd_constructor()
env.generate_dd()
env.compute_pareto_frontier()
z = env.get_frontier()
pprint(z)
