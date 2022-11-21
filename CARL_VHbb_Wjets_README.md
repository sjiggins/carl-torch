# Using CARL in VHbb for W+jets (1L channel)

## Training CARL NN

### Producing training trees
The training MVA trees are produced by the CxAODReader. The most efficient way to do this is to run nominal (Sherpa) and alternative (MadGraph) separately:

**Sherpa**

```bash
$SubmitR -i $cxaod -c 1L -mc {a,d,e} -a VHbb -as Resolved -tr T -tm T -v 33-05 \
-s "WenuB_Sh221 WenuC_Sh221 WenuL_Sh221 Wenu_Sh221 WmunuB_Sh221 WmunuC_Sh221 WmunuL_Sh221 \
Wmunu_Sh221 WtaunuB_Sh221 WtaunuC_Sh221 WtaunuL_Sh221 Wtaunu_Sh221" \
-exec 1 -o <name>_Sh221 -driver condorDAG -eos /eos/user/${USER:0:1}/${USER}/CxAOD_output \
-doMVATraining -condorq workday
```

**MadGraph**

```bash
$SubmitR -i $cxaod -c 1L -mc {a,d,e} -a VHbb -as Resolved -tr T -tm T -v 33-05 \
-s "WenuB_MGPy8 WenuC_MGPy8 WenuL_MGPy8  WmunuB_MGPy8 WmunuC_MGPy8 WmunuL_MGPy8 \
WtaunuB_MGPy8 WtaunuC_MGPy8 WtaunuL_MGPy8" -exec 1 -o <name>_MGPy8 -driver condorDAG \
-eos /eos/user/${USER:0:1}/${USER}/CxAOD_output -doMVATraining -condorq workday
```

In both cases the $cxaod variable must be set to the path to the input CxAOD's, the folder `CxAOD_output`must exist. \<name> is to be replaced by a suitable name.

The training trees can now be hadded and moved into a common folder within the `CxAOD_output` folder:

```bash
mkdir <name>
hadd <name>/Wjets_Sh221.root <name>_Sh221/*/haddedTree/hadded_all.root
hadd <name>/Wjets_MGPy8.root <name>_MGPy8/*/haddedTree/hadded_all.root
```
In order to use the `hadd` command, it is recommended to use the one obtained by running `lsetup root`.

### Training models

The models can be trained with the *CARL py-torch* code ([link](https://github.com/sjiggins/carl-torch)) by the following command:

```bash
python train.py -n Wjets_Sh221 -v Wjets_MGPy8 -e -1 \
-p /eos/user/${USER:0:1}/${USER}/CxAOD_output/<name>/ -g <model_global_name> \
-t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nTags,nJ,FlavourLabel" \
-w EventWeight --scale-method minmax --BoolFilter "nTags==2 & nJ <= 3 & FlavourLabel==2"
```
The variables used in the `BoolFilter` *must* be listed in the input variable list `-f`. The training can be run locally on a CPU but it is recommended to use GPU via the lxplus batch system. This can be easily achieved by having an *HTCondor* submission scrip as follows:

```
# train.sub

executable              = CARLRun.sh
arguments               = $(ClusterId)$(ProcId)
output                  = CARLRun.$(ClusterId).$(ProcId).out
error                   = CARLRun.$(ClusterId).$(ProcId).err
log                     = CARLRun.$(ClusterId).log
should_transfer_files   = YES
+JobFlavour             = "tomorrow"
request_GPUs = 1
request_CPUs = 0
```
The job flavour can be adjusted as required. The executable `CARLRun.sh` will then look like:

```bash
#!/bin/bash

cd /afs/cern.ch/user/[path to the py-torch code]/carl-torch

source CarlEnv/bin/activate

python train.py [arguments as described above]

deactivate

```

### Evaluation using the py-torch code

Regardless of whether the training is run locally or on the batch system, the evaluation must be run locally on CPU. The input arguments are the same as for the training script:

```bash
python train.py -n Wjets_Sh221 -v Wjets_MGPy8 -e -1 \
-p /eos/user/${USER:0:1}/${USER}/CxAOD_output/<name>/ -g <model_global_name> \
-t Nominal -f "MET,dPhiLBmin,dPhiVBB,dRBB,dYWH,mBB,mTW,Mtop,pTB1,pTB2,pTV,nTags,nJ,FlavourLabel" \
-w EventWeight --scale-method minmax
```
The evaluation plots are saved in `plots/<model_global_name>`. These plots are fully inclusive. In order to produce evaluation plots for each region, please refer to the instructions in the following chapter.
## Evaluation CARL performance
This section describes how to use the trained CARL models within the CxAOD Framework and, in particular, how to produce histograms and evaluation plots for all regions.

### Loading the ONNX models into CxAOD Reader
The trained models are saved in `carl-torch/models/*_new.onnx`. They have to be copied into the *CorrsAndSysts* module of the CxAOD Framework. They have to copied into:

```bash
CxAODReaderCore/VHbb/CorrsAndSysts/data/CarlReweighter
```
Before the [MR](https://gitlab.cern.ch/CxAODFramework/CorrsAndSysts/-/tree/dev-lvozdeck-carlReweighter) is closed, the [dev-lvozdeck-carlReweighter](https://gitlab.cern.ch/CxAODFramework/CorrsAndSysts/-/tree/dev-lvozdeck-carlReweighter) branch must be checked out.

At the moment the CxAOD Reader expects 6 models, one for each flavour, with the following names:

* Carl\_Wbb\_SHtoMG5.onnx
* Carl\_Wbc\_SHtoMG5.onnx
* Carl\_Wbl\_SHtoMG5.onnx
* Carl\_Wcc\_SHtoMG5.onnx
* Carl\_Wcl\_SHtoMG5.onnx
* Carl\_Wl\_SHtoMG5.onnx

### Dev-branches of CxAOD Reader modules
The CARL re-weighter is not yet fully incorporated into the CxAOD Framework. In order to produce histograms with CARL systematic variations, the following branches need to be checked out:

* VHbb/CxAODReader_VHbb: [dev-lvozdeck-carlReweighter-truthTagging](https://gitlab.cern.ch/CxAODFramework/CxAODReader_VHbb/-/tree/dev-lvozdeck-carlReweighter-truthTagging)
* VHbb/CorrsAndSysts: [dev-lvozdeck-carlReweighter](https://gitlab.cern.ch/CxAODFramework/CorrsAndSysts/-/tree/dev-lvozdeck-carlReweighter)
* Core/CxAODReader: change describe below

### Saving the statistical errors

By default the CxAOD Framework does not save the statistical errors on the systematic histograms. One needs to tweak the code in `Core/CxAODReader/Root/HistSvc.cxx`

```
if (
    UtilCode::stringStartsWith(fullname,"Sys") && 
    UtilCode::stringContains(fullname,"/")     && 
    !m_systHistSaveStatUncert){
    ...
```

to include an exception for the CARL systematic variation:

```
if (
	! UtilCode::stringContains(fullname,"Carl") &&
    UtilCode::stringStartsWith(fullname,"Sys") && 
    UtilCode::stringContains(fullname,"/")     && 
    !m_systHistSaveStatUncert){
    ...
```

### Producing histograms

After compiling CxAOD Reader (see [here](https://gitlab.cern.ch/CxAODFramework/CxAODReaderCore/-/blob/master/README.md)), the histograms can be produced with the following command:

```bash
$SubmitR -i $cxaod -c 1L -mc {a,d,e} -a VHbb -as Resolved -tr T -tm T -v 33-05 \
-s "WenuB_MGPy8 WenuB_Sh221 WenuC_MGPy8 WenuC_Sh221 WenuL_MGPy8 WenuL_Sh221 Wenu_Sh221 \
WmunuB_MGPy8 WmunuB_Sh221 WmunuC_MGPy8 WmunuC_Sh221 WmunuL_MGPy8 WmunuL_Sh221 Wmunu_Sh221 \
WtaunuB_MGPy8 WtaunuB_Sh221 WtaunuC_MGPy8 WtaunuC_Sh221 WtaunuL_MGPy8 WtaunuL_Sh221 \
Wtaunu_Sh221" -exec 1 -o <histograms_name> -driver condorDAG \
-eos /eos/user/${USER:0:1}/${USER}/CxAOD_output -doSysts -condorq tomorrow -histlvl 1
```
The histograms can then be combined into a single root file:

```bash
cd /eos/user/${USER:0:1}/${USER}/CxAOD_output/<histograms_name>
hadd ade.root */haddedHist/hadded_all.root
```

### Closure plots

The closure plots are produced by a python script that is part of the [repository](https://gitlab.cern.ch/lvozdeck/carlrewighter) containing the stand-alone CARL C++ code (described in greater detail in the next chapter).

```bash
cd carlrewighter/plotting
mkdir plots

lsetup root
python plot.py /eos/user/${USER:0:1}/${USER}/CxAOD_output/<histograms_name>/ade.root
```
The script produces plots in `plotting/plots`. It also produces a text file `plots/closure.csv` with the chi2/n.d.f. values quantifying the closure and a LaTeX table `plots/closure_table.tex` with colour coding.

## CARL C++ stand-alone code
The stand-alone C++ CARL codebase ([link](https://gitlab.cern.ch/lvozdeck/carlrewighter)) enables one to evaluate CARL ONNX models without having to run the cumbersome CxAOD Reader. At the moment the program produces a calibration plot, but can be adapted to anything else.
