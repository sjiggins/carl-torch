if [[ $(uname -a) == *"cent7"* ]]; then
    # centos7
    setupATLAS -q
    export PIP_NO_CACHE_DIR=off
    lsetup "views LCG_100_ATLAS_1 x86_64-centos7-gcc10-opt"
    TEMP_PYTHONPATH=/cvmfs/sft.cern.ch/lcg/views/LCG_100_ATLAS_1/x86_64-centos7-gcc10-opt/bin/python:/cvmfs/sft.cern.ch/lcg/views/LCG_100_ATLAS_1/x86_64-centos7-gcc10-opt/lib
elif [[ $(uname -a) == *"sdf"* ]]; then
    # SLAC SLURM cluster
    setupATLAS -q
    export PIP_NO_CACHE_DIR=off
    lsetup "views LCG_100_ATLAS_1 x86_64-centos7-gcc10-opt"
    TEMP_PYTHONPATH=/cvmfs/sft.cern.ch/lcg/views/LCG_100_ATLAS_1/x86_64-centos7-gcc10-opt/bin/python:/cvmfs/sft.cern.ch/lcg/views/LCG_100_ATLAS_1/x86_64-centos7-gcc10-opt/lib
elif [[ $(uname -a) == *"slc6"* ]]; then
    # slc6
    setupATLAS -q
    export PIP_NO_CACHE_DIR=off
    lsetup "views LCG_98bpython3 x86_64-centos7-gcc8-opt"
    TEMP_PYTHONPATH=/cvmfs/sft.cern.ch/lcg/views/LCG_98bpython3/x86_64-slc6-gcc8-opt/python:/cvmfs/sft.cern.ch/lcg/views/LCG_98bpython3/x86_64-slc6-gcc8-opt/lib
fi
OLD_PYTHONPATH=$PYTHONPATH
if [ -f "CARL-Torch/bin/activate" ]; then
    source CARL-Torch/bin/activate
else
    python -m venv CARL-Torch
    source CARL-Torch/bin/activate
    PYTHONPATH=$TEMP_PYTHONPATH
    python -m pip install -U pip setuptools wheel
    echo 'Now run `python -m pip install -e .'
fi
export PYTHONPATH=$(pwd)/CARL-Torch/lib/python3.8/site-packages:$OLD_PYTHONPATH
