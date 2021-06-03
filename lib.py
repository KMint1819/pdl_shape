from pathlib import Path
import numpy as np

def get_save_dir(d):
    if type(d) is str:
        d = Path(d)
    out = "exp"
    m = 0
    if Path.is_dir(d):
        arr = [-1]
        for p in d.iterdir():
            arr.append(str(p.stem).split('exp')[1])
        arr = np.array(arr, dtype=np.int)
        m = arr.max() + 1
    (d / (out + str(m))).mkdir(parents=True)
    return d / (out + str(m))