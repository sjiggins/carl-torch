#!/bin/bash

echo "jobNumber:"
echo $1

echo "input:"
echo $2

if [ $1 == 1 ]; then
  nominal=V2.Wjets_bb_Sh221_-1
  alternative=V2.Wjets_bb_MGPy8_-1
fi

if [ $1 == 2 ]; then
  nominal=V2.Wjets_bc_Sh221_-1
  alternative=V2.Wjets_bc_MGPy8_-1
fi

if [ $1 == 3 ]; then
  nominal=V2.Wjets_bl_Sh221_-1
  alternative=V2.Wjets_bl_MGPy8_-1
fi

if [ $1 == 4 ]; then
  nominal=V2.Wjets_cc_Sh221_-1
  alternative=V2.Wjets_cc_MGPy8_-1
fi

if [ $1 == 5 ]; then
  nominal=V2.Wjets_cl_Sh221_-1
  alternative=V2.Wjets_cl_MGPy8_-1
fi

if [ $1 == 6 ]; then
  nominal=V2.Wjets_l_Sh221_-1
  alternative=V2.Wjets_l_MGPy8_-1
fi

echo "nominal = $nominal"
echo "alternative = $alternative"

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

python train.py -n $nominal -v $alternative -e -1 -p ../ -g gridTest -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax #--n_workers $CPUs

python evaluate.py -n $nominal -v $alternative -e -1 -p ../ -g gridTest -t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nJ" -w EventWeight --scale-method minmax

tar -cf output.tar data models plots
cp output.tar ../
