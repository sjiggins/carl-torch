#!/usr/bin/python

from __future__ import absolute_import, division, print_function

# Progress Bar
import time
import sys

import random

# Root Imports
import ROOT
# Import root_numpy
#from root_numpy import array2tree

# standard numerical library imports
import numpy as np
import scipy as sp
from scipy.stats import *
import math as math

# Import optimisation from scipy
from scipy import optimize

# Utility for plotting (custom)
import PlotUtils as PU

# Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt

# Global plot settings
from matplotlib import rc
import matplotlib.font_manager
rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

# Seed setting
from numpy.random import seed
seed(12345)
np.random.seed(12345)


##########################
# Monte Carlo Integrator #
##########################
def ValidResonance(mab,width,mr):
    #    BR = 1 / mr**2 - mab**2 - i*width

    # Probability of resonance
    prob = sp.stats.cauchy.pdf(mab,loc=mr,scale=width)

    # Generate random cauchy number
    #prob = stats.cauchy.rvs(loc=mr,scale=width,size=1)

    # Now generate a random number from a uniform
    uni = np.random.uniform(0,sp.stats.cauchy.pdf(mr,loc=mr,scale=width),1)

    # Check if successful otherwise veto
    if uni > prob:
        return -1
    else:
        return 1
    
    
    

def Mass(vec = []):
    mass = vec[0]**2 - vec[1]**2 - vec[2]**2 - vec[3]**2
    if mass < 0:
        return 0.0
    else:
        return math.sqrt(mass)

def Angle(Ei,Pi,mi,Ej,Pj,mj,mij):

    nominator = mi**2 + mj**2 - mij**2 + 2*Ei*Ej
    denominator = 2*Pi*Pj
    cos_theta_ij = nominator/denominator
    #print ("cos_theta_ij: {}".format(cos_theta_ij))
    if cos_theta_ij < -1.0 or cos_theta_ij > 1.0:
        return 10.0
    else:
        return math.acos(cos_theta_ij)


def PhaseSpaceSample(Events, Weights, 
                     ROOTfile, treeName, 
                     M, m1, m2, m3, n_points, 
                     randomise=False, POI=None, SubSet=True, BkgRate = 0.5):

    # Check for valid POI
    if POI==None:
        print ("<PhaseSpaceSample>::   POI for event generation not set. Will utilise 0, be warned.")
        POI[0] = 0

    # Define TTree storange
    #E      = np.empty((3), dtype='float64')
    E      = ROOT.std.vector('float')()
    #theta  = np.empty((3), dtype='float64')
    theta      = ROOT.std.vector('float')()
    #momP   = np.empty((3), dtype='float64')
    momP      = ROOT.std.vector('float')()
    #PDGID  = np.empty((3), dtype='float64')
    PDGID      = ROOT.std.vector('float')()
    #mass   = np.empty((3), dtype='float64')
    mass      = ROOT.std.vector('float')()
    #width  = np.empty((3), dtype='float64')
    width      = ROOT.std.vector('float')()
    # Weights
    w          = np.empty((1), dtype='float64')
    # Spectators
    #   m12, m13, m23, m123
    m12_spec   = np.empty((1), dtype='float64')
    m13_spec   = np.empty((1), dtype='float64')
    m23_spec   = np.empty((1), dtype='float64')
    m123       = np.empty((1), dtype='float64')
    tree = ROOT.TTree(treeName, treeName)
    tree.Branch("E",     E)
    tree.Branch("Theta", theta)
    tree.Branch("momP",  momP)#, "vector<float>")
    tree.Branch("PDGID", PDGID)#, "vector<float>")
    tree.Branch("mass",  mass)#, "vector<float>")
    tree.Branch("width", width)#, "vector<float>")
    tree.Branch('w',     w, "w/D")
    tree.Branch('m12',      m12_spec,  "m12/D")
    tree.Branch('m13',      m13_spec,  "m13/D")
    tree.Branch('m23',      m23_spec,  "m23/D")
    tree.Branch('m213',     m123,      "m123/D")

    # Form numpy array of elements
    #   Numpy array shape:
    #        Planes = particles per event (3 in this case)
    #        Rows = event number (1 to seed with)
    #        Columns = (e,px,py,pz) four-vector per particle
    EventsSample = np.zeros((n_points,3,5))
    
    ## Get the integrated estimate of the phase space
    #Integral,IntVar = PhaseSpaceIntegrator( M, m1, m2, m3, n_points)
    #
    #print ("Width:    {}".format(Integral))
    #print ("Error:    {}".format(IntVar))
    
    m12_array = []
    m13_array = []
    m23_array = []
    E3_array = []
    E2_array = []
    E1_array = []
    theta12_array = []
    theta13_array = []
    theta23_array = []
    # Now randomly sample (rejection sampling)
    # Generate possible mass pairings
    #m13 = np.random.uniform( m1+m3, M - m2 , 1)
    n = 0
    toolbar_width = 100
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    while n < n_points:

        E.clear()
        theta.clear()
        momP.clear()
        PDGID.clear()
        mass.clear()
        width.clear()



        # Set POI for start
        tempPOI = [POI[0],POI[2]]
        
        #print ("=====================")
        #print ("=====================")
        m12=-1
        if np.random.uniform(0,1,1) > BkgRate:
            m12 = np.random.uniform( (m1+m2)**2, (M - m3)**2 , 1)
            m12 = math.sqrt(m12[0])
        else:
            m12 = sp.stats.cauchy.rvs(loc=POI[0],scale=POI[2],size=1)
            m12 = m12[0]
        if m12 < (m1+m2) or m12 > (M - m3):
            continue
        m23 = np.random.uniform( (m2+m3)**2, (M - m1)**2 , 1)
        m23 = math.sqrt(m23[0])

        #if ValidResonance(m12,2,tempPOI) == 1:
        #    if m12 < (m1+m2) or m12 > (M - m3):
        #        continue
        #else:
        #    continue
        
        # Determine m23
        m13 = M**2 + m1**2 + m2**2 + m3**2 - m12**2 - m23**2
        #m13 = M**2 + m1**2 + m2**2 + m3**2 - m12[0] - m23[0]
        E2  = (M**2 + m2**2 - m13) / (2*M)
        if m13 < 0 or (E2**2 - m2**2) < 0 or E2 < 0:
            continue
        m13 = math.sqrt(m13)
        #print("-------------------")

        # Energy/mom of particle 3
        #print ("m12:  {}".format(m12))
        E3 = (M**2 + m3**2 - m12**2) / (2*M)
        momP3 = math.sqrt(E3**2 - m3**2)
               
        # Energy/mom of particle 1
        E1 = (M**2 + m1**2 - m23**2) / (2*M)
        momP1 = math.sqrt(E1**2 - m1**2)

        # Energy/mom of particle 2
        momP2 = math.sqrt(E2**2 - m2**2)

        ## Calculate the 3 angles
        theta12 = Angle(E1,momP1,m1,E2,momP2,m2,m12)
        theta13 = Angle(E1,momP1,m1,E3,momP3,m3,m13)
        theta23 = Angle(E2,momP2,m2,E3,momP3,m3,m23)
        
        if theta12 + theta23 + theta13 > 2*math.pi*1.001 :
            continue

        # Energy conservation and on-shell requirements
        if (E1 + E2 + E3)**2 > (M**2)*1.01 or (E1**2 + E2**2 + E3**2)**2 < (M**2)*0.99 or E1 < m1 or E2 < m2 or E3 < m3:
            continue

        m12_array.append(float(m12**2))
        m13_array.append(float(m13**2))
        m23_array.append(float(m23**2))

        theta12_array.append(float(theta12))
        theta13_array.append(float(theta13))
        theta23_array.append(float(theta23))

        E3_array.append(float(E3))
        E2_array.append(float(E2))
        E1_array.append(float(E1))

        # Expand vectors with parameter of interest
        if randomise:
            tempPOI = [random.choice([25.,30.,40.,50.,55.,65.]), random.choice([2.,4,6.,8.,10,12])]
            
        if not SubSet:
            tempPOI = [POI[1],POI[3]]
        vec1 = np.array( [E1, theta13, momP1, 1] + tempPOI ) # 1 = PDG ID
        vec2 = np.array( [E2, theta23, momP2, 2] + tempPOI ) # 2 = PDG ID
        vec3 = np.array( [E3, 0,       momP3, 3] + tempPOI ) # 3 = PDG ID

        # By reference change the input Events Numpy Array
        Events[n,0,:] = vec1
        Events[n,1,:] = vec2
        Events[n,2,:] = vec3
        # Now allocate a weight
        Weights[n] = np.random.uniform( 0.0, 2.0 , 1) # Mean of 1
        # Check for negative weight contribution
        if m12 > 50 and ROOTfile == "ConditionalSet":
            Weights[n] -= (m12/80 - 0.625)
        
        # Now fill the tree
        E.push_back(E1)
        E.push_back(E2)
        E.push_back(E3)

        #theta = np.array( [theta13, theta23, 0] )
        theta.push_back(theta13)
        theta.push_back(theta23)
        theta.push_back(0)
        
        #momP = np.array( [momP1, momP2, momP3] )
        momP.push_back(momP1)
        momP.push_back(momP2)
        momP.push_back(momP3)
        
        #PDGID = np.array( [1, 2, 3] )
        PDGID.push_back(1)
        PDGID.push_back(2)
        PDGID.push_back(3)
        
        #mass = np.array( [POI[1], POI[1], POI[1]] )
        mass.push_back(POI[1])
        mass.push_back(POI[1])
        mass.push_back(POI[1])

        #width = np.array( [POI[3], POI[3], POI[3]] )
        width.push_back(POI[3])
        width.push_back(POI[3])
        width.push_back(POI[3])
        
        # Weights
        w[0] = Weights[n]

        # Spectators
        m12_spec[0] = m12
        m13_spec[0] = m13
        m23_spec[0] = m23
        m123[0] = math.sqrt(  (E1+E2+E3)**2 - (momP1+momP2+momP3)**2 )
        
        tree.Fill()

        # increment
        if n % (n_points/100) == 0:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()
        n += 1

    sys.stdout.write("]\n") # this ends the progress bar
    file = ROOT.TFile(ROOTfile+".root", "RECREATE")
    tree.Write()
    file.Close()

    #if SubSet:
    #    return
    #else:
    #    return EventsSample

    
    print ("PhaseSpaceSampling:: Event Sampling done")
    print ("PhaseSpaceSampling:: Plotting")
    plt.hist2d(m12_array, m23_array, bins = 100, weights=Weights, norm=mpl.colors.LogNorm())
    plt.savefig('CARLDataSet-neg/Dalitz-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    plt.clf()
    plt.close()
    
    plt.hist2d(m13_array, m23_array, bins = 100, weights=Weights, norm=mpl.colors.LogNorm())
    plt.savefig('CARLDataSet-neg/Dalitz-12-23-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    plt.clf()
    plt.close()
    
    plt.hist2d(m12_array, m13_array, bins = 100, weights=Weights, norm=mpl.colors.LogNorm())
    plt.savefig('CARLDataSet-neg/Dalitz-12-13-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    plt.clf()
    plt.close()
    
    #plt.hist2d(theta12_array, theta23_array, bins = 100, norm=mpl.colors.LogNorm())
    #plt.savefig('plots-Info-2ME/theta12-23-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    #plt.hist2d(theta12_array, theta13_array, bins = 100, norm=mpl.colors.LogNorm())
    #plt.savefig('plots-Info-2ME/theta12-13-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    #plt.hist2d(theta13_array, theta23_array, bins = 100, norm=mpl.colors.LogNorm())
    #plt.savefig('plots-Info-2ME/theta13-23-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    ##PU.DataSetPlotFill1D(EventsSample[:,:,0],
    ##                     'Energy(POI = {}) - '.format(POI),
    ##                     'plots-Info-2ME/E-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3))
    #bins = np.linspace(np.amin(EventsSample[:,:,0]), np.amax(EventsSample[:,:,0]), 100)
    #bins = np.linspace(0,100,100)
    #plt.hist(EventsSample[:,0,0], bins = bins, alpha=0.5, label = r'$E1:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,1,0], bins = bins, alpha=0.5, label = r'$E2:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,2,0], bins = bins, alpha=0.5, label = r'$E3:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.legend(loc='upper right', frameon=False, prop={"size":8})
    #plt.savefig('plots-Info-2ME/E-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    #bins = np.linspace(np.amin(EventsSample[:,:,1]), np.amax(EventsSample[:,:,1]), 100)
    #plt.hist(EventsSample[:,0,1], bins = bins, alpha=0.5, label = r'$\theta 13:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,1,1], bins = bins, alpha=0.5, label = r'$\theta 23:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,2,1], bins = bins, alpha=0.5, label = r'$\theta 12:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.legend(loc='upper right', frameon=False, prop={"size":8})
    #plt.savefig('plots-Info-2ME/theta-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    #bins = np.linspace(np.amin(EventsSample[:,:,2]), np.amax(EventsSample[:,:,2]), 100)
    #plt.hist(EventsSample[:,0,2], bins = bins, alpha=0.5, label = r'$\theta 13:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,1,2], bins = bins, alpha=0.5, label = r'$\theta 23:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(EventsSample[:,2,2], bins = bins, alpha=0.5, label = r'$\theta 12:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.legend(loc='upper right', frameon=False, prop={"size":8})
    #plt.savefig('plots-Info-2ME/Pz-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()
    #
    ##bins = np.linspace(np.amin(EventsSample[:,9]), np.amax(EventsSample[:,9]), 100)
    ##plt.hist(EventsSample[:,9], bins = bins, alpha=0.5, label = r'$\theta 12:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    ##plt.hist(EventsSample[:,10], bins = bins, alpha=0.5, label = r'$\theta 13:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    ##plt.hist(EventsSample[:,11], bins = bins, alpha=0.5, label = r'$\theta 23:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    ##plt.legend(loc='upper right', frameon=False, prop={"size":8})
    ##plt.savefig('plots-Info-2ME/Mass-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    ##plt.clf()
    ##plt.close()
    #
    #bins = np.linspace(0,10000,100)
    #plt.hist(m12_array, bins = bins, alpha=0.5, label = r'$m12 sel:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(m13_array, bins = bins, alpha=0.5, label = r'$m13 sel:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.hist(m23_array, bins = bins, alpha=0.3, label = r'$m23 sel:  POI=, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
    #plt.legend(loc='upper right', frameon=False, prop={"size":8})
    #plt.savefig('plots-Info-2ME/mij-PFN_{}_{}_{}_{}.pdf'.format(POI,m1,m2,m3),format='pdf')
    #plt.clf()
    #plt.close()    

    if SubSet:
        return
    else:
        #return EventsSample
        return m12_array


#######################
# Reweighting example #
#######################

LoadData=False
inputDims=6

# --------------
# Build datasets
# --------------

## Generate ROOT TTree
obs = ['E', 'Theta', 'momP', 'PDGID', 'Mass', 'width']
n_data_points = 300000 

# Parameter defaults
POI = [40.,40.,6.,6.]
M = 100.
m1 = 10.
m2 = 10.
m3 = 10.

# Conditional mapping set
#GridSize = 36 # Attempt 1
#GridPoints_m1 = [25.,30.,40.,50.,55.,65.] # Attempt 1
#GridPoints_width1 = [2.,4.,6.,8.,10.,12] # Attempt 1

GridSize = 1 # Attempt 3
GridPoints_m1 = [50.] # Attempt 3
GridPoints_width1 = [6.] # Attempt 3


# Data set
ConditionalSet = np.zeros((1,3,inputDims))
ConditionalSet_weights = np.zeros((1))

# Default
Default =  np.zeros((n_data_points*GridSize,3,inputDims))
Default_weights = np.zeros((n_data_points*GridSize))
if LoadData:
    Default = np.load("ThreeBody_Default_CARL.npz")
    Default_weights = np.load("ThreeBody_Default_CARL_weights.npz")
else:
    PhaseSpaceSample(Default, Default_weights, 
                     "DefaultSet", "M50_6",
                     M, m1, m2, m3, 
                     n_data_points*GridSize, True, POI, True, 0.5)
    np.save("ThreeBody_Default_CARL", Default)
    np.save("ThreeBody_Default_CARL_weights", Default_weights)



# Generate the Grid points
if LoadData:
    ConditionalSet = np.load("ThreeBody_Cond_CARL.npz")
    ConditionalSet_weights = np.load("ThreeBody_Cond_CARL_weights.npz")
else:
    for p_m in GridPoints_m1:
        for p_w in GridPoints_width1:
            Events = np.zeros((n_data_points,3,inputDims))
            Weights = np.zeros((n_data_points))
            PhaseSpaceSample( Events, Weights,  
                              "ConditionalSet", "M50_6",
                              M, m1, m2, m3, 
                              n_data_points, False, [p_m,p_m,p_w,p_w],True, 0.5)
            ConditionalSet = np.vstack( (ConditionalSet,Events) )
            ConditionalSet_weights = np.append( ConditionalSet_weights, Weights, axis=0 )
            
    ConditionalSet = np.delete(ConditionalSet, (0), axis=0)
    ConditionalSet_weights = np.delete(ConditionalSet_weights, (0), axis=0)
    np.save("ThreeBody_Cond_CARL", ConditionalSet)
    np.save("ThreeBody_Cond_CARL_weights", ConditionalSet_weights)


# Print shape of conditional and default sets
print ("=== Default ===")
print ("   -> X:  {}".format(Default.shape))
print ("   -> W:  {}".format(Default_weights.shape))
print ("=== Conditional Set ===")
print ("   -> X: {}".format(ConditionalSet.shape))
print ("   -> W: {}".format(ConditionalSet_weights.shape))

#Generate diagnostic plots-Info-2ME
bins = np.linspace(np.amin(Default[:,:,3]),np.amax(Default[:,:,3]),100)
fig, (ax1,ax2) = plt.subplots(nrows=2)
print("Plotting")
h0 = ax1.hist(Default[:,0,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Default:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h1 = ax1.hist(Default[:,1,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Default:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h2 = ax1.hist(Default[:,2,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Default:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h1 = ax2.hist(ConditionalSet[:,0,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Cond.  :  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,1,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Cond.  :  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,2,3], bins = 50, alpha = 0.5, weights=Default_weights, label = r'$Cond.  :  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
ax1.legend(loc='upper right', frameon=False, prop={"size":8})
ax2.legend(loc='upper right', frameon=False, prop={"size":8})
plt.savefig('CARLDataSet-neg/PFN_POISummary.pdf',format='pdf')
plt.clf()
plt.close()


bins = np.linspace(np.amin(Default[:,:,0]),np.amax(Default[:,:,0]),100)
fig, (ax1,ax2) = plt.subplots(nrows=2)
print("Plotting")
h0 = ax1.hist(Default[:,0,0], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h0 = ax1.hist(Default[:,1,0], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h0 = ax1.hist(Default[:,2,0], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h1 = ax2.hist(ConditionalSet[:,0,0], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,1,0], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,2,0], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
ax1.legend(loc='upper right', frameon=False, prop={"size":8})
ax2.legend(loc='upper right', frameon=False, prop={"size":8})
plt.savefig('CARLDataSet-neg/PFN_E-Summary.pdf',format='pdf')
plt.clf()
plt.close()

bins = np.linspace(np.amin(Default[:,:,1]),np.amax(Default[:,:,1]),100)
fig, (ax1,ax2) = plt.subplots(nrows=2)
print("Plotting")
h0 = ax1.hist(Default[:,0,1], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h0 = ax1.hist(Default[:,1,1], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
#h0 = ax1.hist(Default[:,2,1], bins = bins, alpha = 0.5,weights=Default_weights,  label = r'$Default P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h1 = ax2.hist(ConditionalSet[:,0,1], bins = bins, alpha = 0.5,  weights=ConditionalSet_weights, label = r'$Cond.  P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,1,1], bins = bins, alpha = 0.5,  weights=ConditionalSet_weights, label = r'$Cond.  P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
#h1 = ax2.hist(ConditionalSet[:,2,1], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
ax1.legend(loc='upper right', frameon=False, prop={"size":8})
ax2.legend(loc='upper right', frameon=False, prop={"size":8})
plt.savefig('CARLDataSet-neg/PFN_Theta-Summary.pdf',format='pdf')
plt.clf()
plt.close()

bins = np.linspace(np.amin(Default[:,:,2]),np.amax(Default[:,:,2]),100)
fig, (ax1,ax2) = plt.subplots(nrows=2)
print("Plotting")
h0 = ax1.hist(Default[:,0,2], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default  P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h0 = ax1.hist(Default[:,1,2], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default  P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h0 = ax1.hist(Default[:,2,2], bins = bins, alpha = 0.5, weights=Default_weights, label = r'$Default  P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,m1,m2,m3))
h1 = ax2.hist(ConditionalSet[:,0,2], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P1:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,1,2], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P2:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
h1 = ax2.hist(ConditionalSet[:,2,2], bins = bins, alpha = 0.5, weights=ConditionalSet_weights, label = r'$Cond.  P3:  POI={}, m1={}, m2={}, m3={}$'.format(POI,"N/A","N/A","N/A"))
ax1.legend(loc='upper right', frameon=False, prop={"size":8})
ax2.legend(loc='upper right', frameon=False, prop={"size":8})
plt.savefig('CARLDataSet-neg/PFN_Pz-Summary.pdf',format='pdf')
plt.clf()
plt.close()


## -----------------------------
print("=============================================")
print("Completed generation")
