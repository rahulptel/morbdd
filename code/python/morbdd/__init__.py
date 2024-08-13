import os
from pathlib import Path
import sys

root_path_dict = {
    "desktop": Path("/home/rahul/Documents/projects/MORBDD/resources"),
    "laptop": Path("/home/rahul/Documents/PhD/projects/MORBDD/resources"),
    "cc": Path("/home/rahulpat/scratch/l2o_resources"),
}
machine = os.environ.get("machine")
assert machine is not None

resource_path = root_path_dict[machine]


class ResourcePaths:
    resource = root_path_dict[machine]
    inst = resource / "instances"
    bdd = resource / "bdds"
    sol = resource / "sols"
    order = resource / "orders"
    dataset = resource / "datasets"
    checkpoint = resource / "checkpoint"
    bin = resource / "bin"


sys.path.append(str(ResourcePaths.bin))


class CONST:
    TIME_COMPILE = 1
    TIME_PARETO = 2
    ONE_ARC = 1
    ZERO_ARC = -1
    RESTRICT = 1
    RELAX = 2
