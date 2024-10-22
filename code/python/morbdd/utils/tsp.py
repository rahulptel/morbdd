import zipfile
import numpy as np
import io

def get_env(n_objs=3):
    modname = "libtspenvv2o" + str(n_objs)
    libddenv = __import__(modname)
    env = libddenv.TSPEnv()

    return env

def get_instance_data(path, size, split, pid, seed=7):
    archive = path.inst / f"tsp/{size}.zip"
    inst = f'{size}/{split}/tsp_{seed}_{size}_{pid}.npz'

    # Open the zip file
    with zipfile.ZipFile(archive, 'r') as z:
        # Open the .npz file from the zip and load it into numpy
        with z.open(inst) as npz_file:
            # Load the .npz content
            data = np.load(io.BytesIO(npz_file.read()))

    return data