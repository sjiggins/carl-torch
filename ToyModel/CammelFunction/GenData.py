import ROOT
import numpy as np
import argparse

# Helper function to create an example tree
def make_data(nevts = 500000, DataType="neg", MC="Nominal"):

    root_file = ROOT.TFile(MC+".root", "RECREATE")
    tree = ROOT.TTree("ThreeDGauss", "ThreeDGauss")
    
    # Input features
    x = np.empty((1), dtype="float32")
    y = np.empty((1), dtype="float32")
    z = np.empty((1), dtype="float32")
    # Event Weights
    w = np.empty((1), dtype="float32")
    # Toggle of gaussian modes
    t0 = np.empty((1), dtype="float32")
    t1 = np.empty((1), dtype="float32")

    # Add features and weights to tree
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "/F")
    tree.Branch("z", z, "z/F")
    tree.Branch("w", w, "w/F")
    tree.Branch("t0", t0, "t0/F")
    tree.Branch("t1", t1, "t1/F")
 
    # Random generate features and weights
    for i in range(nevts):
 
        ## Negative weight version
        if( DataType == "neg"):
            w[0] =  (np.ceil( np.random.random()  - 0.1 ) * 4) - 2  if MC=="Alt" else 2.0
            xmu, xsigma = 0.0, 1.5 if w[0] > 0 else  0.25
            ymu, ysigma = 0.0, 1.5 if w[0] > 0 else  0.25
            zmu, zsigma = 0.0, 1.5 if w[0] > 0 else  0.25

            x[0] = np.random.normal(xmu, xsigma, 1)
            y[0] = np.random.normal(ymu, ysigma, 1)
            z[0] = np.random.normal(zmu, zsigma, 1)
        elif (DataType == "pos"):
            ## Composite positive weight version - gaussian mixture model
            t0[0] = 0.0 if MC=="Nominal" else np.ceil( np.random.random()  - 0.3 )
            t1[0] = 0.0 if MC=="Nominal" else np.ceil( np.random.random()  - 0.5 )
            w[0] = ( ( np.random.random() ) * 2) # Alt + Nominal = uniform sampling with mean = 1.0
        
            # x-axis parameters
            xmu, xsigma = (0.0, 1.0) if MC == "Nominal" else (0.0, 0.6)
            xmu_dis = 0.0 if MC == "Nominal" else  0.9 
            xmu_neg, xsigma_neg = 0.0, 0.5
            # y-axis parameters
            ymu, ysigma = (0.0, 1.0) if MC == "Nominal" else (0.0, 0.6)
            ymu_dis = 0.0 if MC == "Nominal" else  0.9
            ymu_neg, ysigma_neg = 0.0, 0.5
            # z-axis paramters
            zmu, zsigma = (0.0, 1.0) if MC == "Nominal" else (0.0, 0.6)
            zmu_dis = 0.0 if MC == "Nominal" else  0.9
            zmu_neg, zsigma_neg = 0.0, 0.5
            
            #x[0] = t0[0]*t1[0]*np.random.normal(xmu-xmu_dis, xsigma, 1) + t0[0]*(1-t1[0])*np.random.normal(xmu+xmu_dis, xsigma, 1) #+ (1-t0[0])*np.random.normal(xmu_neg, xsigma_neg, 1)
            x[0] = t1[0]*np.random.normal(xmu-xmu_dis, xsigma, 1) + (1-t1[0])*np.random.normal(xmu+xmu_dis, xsigma, 1) #+ (1-t0[0])*np.random.normal(xmu_neg, xsigma_neg, 1)
            y[0] = t1[0]*np.random.normal(ymu-ymu_dis, ysigma, 1) + (1-t1[0])*np.random.normal(ymu+ymu_dis, ysigma, 1) #+ (1-t0[0])*np.random.normal(ymu_neg, ysigma_neg, 1)
            z[0] = t1[0]*np.random.normal(zmu-zmu_dis, zsigma, 1) + (1-t1[0])*np.random.normal(zmu+zmu_dis, zsigma, 1) #+ (1-t0[0])*np.random.normal(zmu_neg, zsigma_neg, 1)



        tree.Fill()
    root_file.Write()
 
    return (root_file, x, y, z, w), tree

#def plot_data():
    

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Specify the type if data you want: [positive definite, negative weigthed]")
    parser.add_argument("--CammelType", type=str, default = "neg", help="Positive definite or negative weighted - [pos,neg]")
    parser.add_argument("--MC", type=str, default = "Nominal", help="Choose between nominal or alternative MC - [Nominal, Alt]")
    parser.add_argument("--nEvents", type=int, default = 500000, help="Number of events")
    args = parser.parse_args()
    
    # Data creation
    _, tree = make_data(nevts=args.nEvents, DataType=args.CammelType, MC=args.MC)
    
if __name__ == "__main__":
    main()
