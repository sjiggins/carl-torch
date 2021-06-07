#!/bin/bash

########################################################
##   Example of HTCondor GPU utilisation
##     https://batchdocs.web.cern.ch/tutorial/exercise10.html
##       https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud
########################################################

## Source python3 version
#source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_97python3 x86_64-centos7-gcc8-opt
#
## Potentially need to build a python virtual environment
##python -m virtualenv myvenv
##source myvenv/bin/activate
##pip install <package 1>
#
## Setup ATLAS environment aliases incase of future need
### Test currn working directory
#test "$TMPDIR" == "" && TMPDIR=/tmp
#
## An example of how to read a 'segments' file that contains names for sub-jobs
#EL_JOBSEG="test-1"
##EL_JOBSEG=`grep "^$EL_JOBID " "/afs/cern.ch/work/s/sjiggins/SM-Vhbb_CxAODFramework/CxAODFramework_branch_master_04.02.20_Tag-r32-24-Reader-Resolved-05_VHbbAllBDTVarSignal-Backup/run/LimitHistograms_mc16ade_VHbbSignal-UEPS-Uncert_012Lepton/Reader_1L_32-15_a_MVA_H/submit/segments" | awk ' { print $2 }'`
## Test Job Segment ID
##test "$EL_JOBSEG" != "" || abortJob
#
## Obtain node host name
#hostname
## Print current location
#pwd
## Test the node identifty - Who Am I !!!
#whoami
#
## ATLAS LOCAL ROOT BASE alias/environment variables
#export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase && source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#RUNDIR=${TMPDIR}/Worker-$EL_JOBSEG-`date +%s`-$$
#mkdir "$RUNDIR" || abortJob
#cd "$RUNDIR" || abortJob
#
## ATLAS setup environment variables
#export AtlasSetupSite=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/AtlasSetup/.config/.asetup.site
#export AtlasSetup=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64/AtlasSetup/V02-00-02/AtlasSetup
#export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#
## Copy over tarball and unpack
#cp /afs/cern.ch/work/s/sjiggins/CARL_Athena/carl-torch.tar.gz .
#tar -xvf carl-torch.tar.gz
#
## Execute python script for
#python3 train.py -s dilepton -v ckkw -n 500000






tar xvf carl-torch-aMC.tar.gz

source CARLENV/bin/activate

python3 train.py --global_name aMCatNLO  --TreeName MVA_Var_Tree_Sig -n Nominal_2jet_bb  -v amcat_2jet_bb -p /afs/cern.ch/work/s/sjiggins/CARL_Athena/carl-torch/Inputs/TTbar_amcatNLO_VHbb/  --features mTop,pTV,mBB,dYWH,dRBB,FlavourLabel --weightFeature EventWeight   -e 1000000

tar cvf models_out.tar.gz models/

deactivate
