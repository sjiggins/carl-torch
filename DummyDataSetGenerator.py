import argparse
import numpy as np
from tempfile import TemporaryFile

# Argument parsing
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nevents', dest='nevents', type=int, action='store', default=-1,
                    help='Define the number of events run over')
parser.add_argument('--set', type=str, dest='set', action='store', default="train",
                    help='Define the set as \'train, val, test\'')
parser.add_argument('--global_name', type=str, dest='global_name', action='store', default="test",
                    help='Define the global name of the training that was done')
args = parser.parse_args()


# Formulate the x-feature array
x = np.array( [[6.901829, 143.427185, 1.852949, 3.084115, 0.614932, 0.621882, 70.177620, 40.955818, 3.0, 2.0, 265.392456, 33.026028, 100.427750]] )
print("Formed the array x: {}".format(x))
print("   -> Shape: {}".format(x.shape))

## Formulate the y array
y = np.array( [0] * x.shape[0] )  # 0 = MC A,   1 = MC B
print("Formed the array y: {}".format(y))
print("   -> Shape: {}".format(y.shape))

# Now need to formulate an array for the weights
w = np.array( [1.0] * x.shape[0] )  # Equal to the number of rows
print("Formed array of weight 'w': {}".format(w))
print("   -> Shape: {}".format(w.shape))

# Save the files to numpy file format
print(args.global_name+"/X0_"+args.set+"_"+str(args.nevents)+".npy")
np.save(args.global_name+"/X0_"+args.set+"_"+str(args.nevents)+".npy", x)
np.save(args.global_name+"/X_"+args.set+"_"+str(args.nevents)+".npy", x)   # Raw duplication as we do not care at this point
print(args.global_name+"/y_"+args.set+"_"+str(args.nevents)+".npy")
np.save(args.global_name+"/y_"+args.set+"_"+str(args.nevents)+".npy", y)
print(args.global_name+"/w0_"+args.set+"_"+str(args.nevents)+".npy")
np.save(args.global_name+"/w0_"+args.set+"_"+str(args.nevents)+".npy", w)
