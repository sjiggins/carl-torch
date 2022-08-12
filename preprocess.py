import argparse

parser = argparse.ArgumentParser(description="Preprocess events.")
parser.add_argument("path", metavar="p", type=str,
                    help="Path to the ROOT files")
parser.add_argument("features", type=str, help="Comma-separated features to select")
parser.add_argument("-n", "--n_events", type=int,
                    help="The number of entries to randomly select (-1 for all)",
                    default=-1)
parser.add_argument("-s","--selection", type=str, help="Event selection", default=None)
parser.add_argument("-o", "--output_path", type=str,
                    help="Output path together with the output ROOT file name.")

args = parser.parse_args()

import uproot

if uproot.__version__ < '4.3.3':
    print("uproot version >= 4.3.3 required! Exiting...")
    exit()

import glob
import pandas as pd
import numpy as np


def totalNumberOfEvents(path):
    nEvents = 0
    for file in glob.glob(path):
        with uproot.open(file) as f:
            nEvents += f["Nominal"].num_entries
    return nEvents

def loadFractionOfEvents(path, features, selection, fraction=1.0):
    allEvents = pd.DataFrame(columns=features, dtype=np.float64)
    for file in glob.glob(path):
        print(file)
        with uproot.open(file)["Nominal"] as tree:
            nEventsToLoad = int(fraction * tree.num_entries) + 1
            df = tree.arrays(features, library="pd", cut=selection)
            if nEventsToLoad > df.shape[0]:
                nEventsToLoad = df.shape[0]
            df = df.sample(n=nEventsToLoad, random_state=42)
            allEvents = pd.concat([allEvents, df], ignore_index=True)
    return allEvents


def main(args):
    print("Counting the total number of events")
    totalNevents = totalNumberOfEvents(args.path)
    print(f"Total number of events: {totalNevents}")
    print(f"selecting events = {args.n_events}")
    if args.n_events == -1:
        fraction = 1.0
    else:
        fraction = args.n_events / totalNevents
        if fraction > 1.0:
            fraction = -1.0
    print(f"fraction of events = {fraction}")
    features = args.features.split(",")
    print(f"features = {features}")
    df = loadFractionOfEvents(args.path, features, args.selection, fraction=fraction)
    print(df)

    outputFile = uproot.recreate(args.output_path)
    outputFile["Nominal"] = df
    
    
if __name__ == "__main__":
    main(args)



