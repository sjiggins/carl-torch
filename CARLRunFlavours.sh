#!/bin/bash

cd /afs/cern.ch/user/l/lvozdeck/public/CarlTorch/carl-torch # change this to your path !!!!

echo "training $3_$2 (FlavourLabel==$1)"

source CarlEnv/bin/activate

python train.py -n Wjets_Sh221_$2 -v Wjets_MGPy8_$2 -e -1 -p ../trainingData/ -g $3_$2 -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax --clipFeatures "mBB,MET,mTW,Mtop,pTB1,pTB2,pTV" --clippingQuantile 0.99

deactivate
