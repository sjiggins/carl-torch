#!/bin/bash
function setup_carl () {
  export AFS_HOME=/afs/cern.ch/user/y/yuzhan
  source $AFS_HOME/carl-torch-master/CARL-Torch/bin/activate

  export NTUPLE_PATH=/eos/user/y/yuzhan/ML_Data/ML_v5/

  export NOMINAL=Zee_Sh2211_Nominal_small
  export VARIATION=Zee_Sh2211_CKKW15
  export TAG=CKKW15
  export TTREE=Zjets
  export WEIGHT=eventWeight
  export NEVENTS=5000000
  export SCALE_METHOD="minmax"
  export FEATURES="matrix_VpT,matrix_Vmass,matrix_Veta,matrix_Jet_Pt,matrix_Njets"
  export FILTER="nLeptons==2"

  export PARENT_DIR=no_polarity_5M_v2
  export SUB_DIR=CKKW15_allJets_matrix

  export THIS_DIR=$AFS_HOME/Training/testBranch_statistic_test/${PARENT_DIR}/${SUB_DIR}/

  export OUTPUT=/eos/user/y/yuzhan/CARLTraining/testBranch_statistic_test/${PARENT_DIR}/${SUB_DIR}/

  mkdir -p $OUTPUT
}

function run_train () {
  python $AFS_HOME/carl-torch-master/carl-torch/train.py \
    -p $NTUPLE_PATH \
    -n $NOMINAL \
    -v $VARIATION \
    -g $TAG \
    -t $TTREE \
    -w $WEIGHT \
    -e $NEVENTS \
    -b $THIS_DIR/binning.yml \
    -f $FEATURES \
    --scale-method $SCALE_METHOD \
    --batch 5000 \
    --layers 10 10 10 10 10 \
    --nepoch 500 \
    --BoolFilter "$FILTER" \
    --spectators "matrix_Lepton_Pt,matrix_Lepton_Eta,matrix_Lepton_Mass,matrix_HT,Lepton_Pt,Lepton_Eta,HT,VpT,Njets" ;
    #--per-epoch-stats ;
    #--per-epoch-plot \
}

function run_evaluate () {
  python $AFS_HOME/carl-torch-master/carl-torch/evaluate.py \
    -p $NTUPLE_PATH \
    -n $NOMINAL \
    -v $VARIATION \
    -g $TAG \
    -t $TTREE \
    -w $WEIGHT \
    -e $NEVENTS \
    -b $THIS_DIR/binning.yml \
    -f $FEATURES \
    --scale-method $SCALE_METHOD \
    --PlotResampleRatio ;
    #--PlotROC \
    #--PlotObsROC \
    #-o "${OUTPUT}" \
}

function exe_carl () {
  # call for setup carl environment and path
  setup_carl

  # default run train and evaluate
  if [ $# -eq 0 ]; then
    run_train
    cp -r * $OUTPUT/

    run_evaluate
    cp -r * $OUTPUT/
  elif [ "$1" = "train" ]; then
    # this only train the model wihout eveluate
    run_train
    cp -r * $OUTPUT/
  elif [ "$1" = "evaluate" ]; then
    # this require existing trained result in the target directory
    # copy everything to current working directory, necessary in condor batch
    cp -r $OUTPUT/data .
    cp -r $OUTPUT/models .
    cp -r $OUTPUT/plots .
    run_evaluate
    cp -r * $OUTPUT/
  else
    echo "Unknown argument $1"
  fi
}

exe_carl "evaluate"
