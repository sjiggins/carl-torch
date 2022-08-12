#!/bin/bash

echo "input:"
echo $1

CPUs=$(nproc --all)
echo "number of CPUs:"
echo $CPUs

if [ -d /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase ]; then
    export ALRB_localConfigDir="/etc/hepix/sh/GROUP/zp/alrb";
    export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase;
    source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh;
else
    \echo "Error: cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase is unavailable" 1>&2;
    return 64;
fi

lsetup git
lsetup "python 3.8.13-fix1-x86_64-centos7"
git clone https://github.com/sjiggins/carl-torch
cd carl-torch
git checkout dev-preprocessing-script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


mkdir data/
mkdir models/
mkdir plots/

python train.py -n Wjets_l_Sh221_5M -v Wjets_l_MGPy8_5M -e -1 -p ../ -g gridTest -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax --n_workers $CPUs

python evaluate.py -n Wjets_l_Sh221_5M -v Wjets_l_MGPy8_5M -e -1 -p ../ -g gridTest -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax

tar -cf output.tar data models plots
cp output.tar ../
