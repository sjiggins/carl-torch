import numpy as np
import uproot
import sys
import pathlib

print("This is the python script.")
print("Numpy array:")
x = np.array([1,2,3,4,5,6])
print(x)

files = sys.argv[1].split(",")

uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource

for file in files:
    print(f"opening file: {file}")
    with uproot.open(pathlib.Path(file))["Nominal"] as tree:
        print(f"{file}: {tree.num_entries}")
