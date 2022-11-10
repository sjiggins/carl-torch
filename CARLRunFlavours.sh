#!/bin/bash

cuts="pTB2<2000"
clipping=""
epochs=300

#selective clipping:
#if [ $2 == "l" ]; then
#  echo "light events => apply clipping"
#  clipping="--clipFeatures \"mBB,Mtop,pTB2\" --clippingQuantile 0.99"
#fi

#selective number of epochs
#if [ $2 == "bb" ]; then
#  epochs=2000
#fi
#
#if [ $2 == "bc" ]; then
#  epochs=2000
#fi
#
#if [ $2 == "cc" ]; then
#  epochs=2000
#fi
#
#if [ $2 == "bl" ]; then
#  epochs=600
#fi
#
#if [ $2 == "cl" ]; then
#  epochs=300
#fi
#
#if [ $2 == "l" ]; then
#  epochs=300
#fi

cd /afs/cern.ch/user/l/lvozdeck/public/CarlTorch/carl-torch # change this to your path !!!!

echo "training $3_$2 (FlavourLabel==$1)"

source CarlEnv/bin/activate

python train.py -n Wjets_Sh221_$2 -v Wjets_MGPy8_$2 -e 5000000 -p ../trainingData/ -g $3_$2 -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax --BoolFilter $cuts $clipping --nepoch $epochs

deactivate
