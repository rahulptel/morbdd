import os
import sys
from pathlib import Path

root_path_dict = {
    "desktop": Path("/home/rahul/Documents/projects/MORBDD/resources"),
    "laptop": Path("/home/rahul/Documents/PhD/projects/MORBDD/resources"),
    "cc": Path("/home/rahulpat/scratch/l2o_resources"),
    "fury": Path("/home/rahul/Documents/phd/MORBDD/resources"),
}
machine = os.environ.get("machine")
assert machine is not None


class ResourcePaths:
    resource = root_path_dict.get(machine)
    inst = resource / "instances"
    bdd = resource / "bdds"
    sol = resource / "sols"
    restricted_sol = resource / "restricted_sols"
    order = resource / "orders"
    dataset = resource / "datasets"
    checkpoint = resource / "checkpoint"
    pretrained = resource / "pretrained"
    bin = resource / "bin"


sys.path.append(str(ResourcePaths.bin))
