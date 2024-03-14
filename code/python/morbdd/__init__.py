import os
from pathlib import Path

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
