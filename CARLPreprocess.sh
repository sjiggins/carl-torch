#!/bin/bash


cd /users/vozdecky/Carl/carl-torch # change this to your path !!!!

ls -l

echo "training flavour $2 (FlavourLabel==$1)"

source CarlEnv/bin/activate

which python

which python3

python3 preprocess.py /data/vozdecky/carlTrainingTrees4_truthTagging/Wjets_Sh221.root "EventWeight,MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -n 5000000 -s "(nTags==2) & (nJ <= 3) & (FlavourLabel==$1)" -o /data/vozdecky/CarlTrainingMVA_prefiltered/Wjets_$2.root

deactivate
