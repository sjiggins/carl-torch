#!/bin/bash

cd /afs/cern.ch/user/l/lvozdeck/public/CarlTorch/carl-torch # change this to your path !!!!

echo "training flavour $2 (FlavourLabel==$1)"

source CarlEnv/bin/activate

python preprocess.py "/eos/user/l/lvozdeck/CxAOD_output/carlTrainingTrees4_truthTagging_Sh221/Reader_1L_33-05_*/haddedTree/tree-*.root" "EventWeight,MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ,FlavourLabel" -n -1 -s "(nTags==2) & (nJ <= 3) & (FlavourLabel==$1)" -o /afs/cern.ch/user/l/lvozdeck/eos/CarlSeparateTrainingTrees/Wjets_Sh221_$2.root

python preprocess.py "/eos/user/l/lvozdeck/CxAOD_output/carlTrainingTrees4_truthTagging_MGPy8/Reader_1L_33-05_*/haddedTree/tree-*.root" "EventWeight,MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ,FlavourLabel" -n -1 -s "(nTags==2) & (nJ <= 3) & (FlavourLabel==$1)" -o /afs/cern.ch/user/l/lvozdeck/eos/CarlSeparateTrainingTrees/Wjets_MGPy8_$2.root

deactivate
