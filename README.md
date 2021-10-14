CARL-TORCH
==================================
[![DOI](https://zenodo.org/badge/255859123.svg)](https://zenodo.org/badge/latestdoi/255859123)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](http://img.shields.io/badge/arXiv-1506.02169-B31B1B.svg)](https://arxiv.org/abs/1506.02169)
[![Build Status](https://travis-ci.com/leonoravesterbacka/carl-torch.svg?branch=master)](https://travis-ci.com/leonoravesterbacka/carl-torch)

*Work in progress by Leonora Vesterbacka (NYU) and Stephen Jiggins (U Freiburg), with input from Johann Brehmer (NYU), Kyle Cranmer (NYU) and Gilles Louppe (U Liège)*

## Introduction
`carl-torch` is a toolbox for multivariate reweighting using PyTorch. 
`carl-torch` is based on [carl](https://github.com/diana-hep/carl/) originally developed for likelihood ratio estimation, but repurposed to be used as a multivariate reweighting technique to be used in the context of particle physics research. 

A more details website for installation, setup, execution, and the theory behind the technique can be found at the public webpage here - [ATLAS Monte Carlo Machine Learning Assisted Uncertainties Forum](https://sjiggins.web.cern.ch/sjiggins/CARL-Webpage/public/)

## Background
The motivation behind this toolbox is to enable a reweighting of one simulation to look like another. 
In the context of particle physics, the research is done largely using simulated data (Monte-Carlo samples). 
As computing resources are finite, effort is put into minimizing the number of simulated samples to be generated, and make the most use of the ones that have to be generated.  

Searches for new physics and measurements of known Standard Model processes are relying on Monte-Carlo samples generated with multiple theoretical settings in order to evaluate their effect on the modelling of the physics process. As it is computationally impractical to generate samples for all theoretical settings with full statistical power, smaller samples are generated for each variation, and a weight is derived to reweight a nominal sample of full statistical power to look like a sample generated with a variational setting. The hope is that the derived weight is able to emulate the effect of the theoretical variation without having to generate it with full statistical power.
### Traditional reweighting in 2 dimensions 
A naive approach to reweighting a sample `p0(x)` to look like another `p1(x)` is by calculating a weight defined as `r(x) = p1(x) / p0(x)` parameterized in one or two dimensions, i.e. as a function of one or two physical variables, and apply this weight to the nominal sample `p0(x)`. By definition, there will be perfect closure when applying the weight to one of the two variables used to calculate the weight. The problem arise when examining the application of the weight to the rest of the variables, where the closure is far from guaranteed. As the disagreement in the other variables will result in large systematic errors, there is a clear motivation to move to a method that can derive the weights using the full phase space. 
### Multivariate reweighting
In order to capture the effects in the full phase space, a *multivariate* reweighting technique is proposed which can take into account n dimensions instead of just two. The technique utilizes a binary classifier trained to differentiate the sample `p0(x)` from sample `p1(x)`, where `x` is an n-dimensional feature vector. 
An ideal binary classifier will estimate `s(x) = p0(x) / (p0(x) + p1(x))` (i.e. the output will be `[0,1]`), and by identifying the weight `r(x) = p1(x) / p0(x)`, the output of the classifier can be rewritten as `s(x) = r(x) / (1 + r(x))`. 
The actual weight `r(x)` is retrieved after expressing `r(x)` as a function of `s(x)`: `r(x) ~ s(x) / (1 - s(x))`. The weights derived using this method are called CARL weights, referring to the name of the method originally proposed in [http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169). Three arbitrary variables (out of n possible ones that can be used in the training) are shown below, for the two distributions `p0(x)` and `p1(x)`. 
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/1_nominalVsVar_1000000.png" width="260">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/2_nominalVsVar_1000000.png" width="260">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/4_nominalVsVar_1000000.png" width="260">
</p>

The classification is done using a PyTorch fully connected DNN, with Adam optimizer and relu activation function, and the calibration of the classifier is done using histogram or isotonic regression. Other optimizers and activation functions are available.  
Once the weights have been calculated as a function of the classifier output `r(x) ~ s(x) / (1 - s(x))`, they are applied to the original sample `p0(x)` (orange histogram), and will ideally line up with the `p1(x)` sample (dashed histogram), for the three arbitrary variables (out of n) that was used in the training. 
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_1_nominalVsVar_train_1000000_fix.png" width="260">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_5_nominalVsVar_train_1000000.png" width="260">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/w_4_nominalVsVar_train_1000000.png" width="260">
</p>

### Performance
The performance of the weights, i.e. how well the reweighted original sample matches the target one, is assessed by training another classifier to discriminate the original sample with weights applied from a target sample. The metric to assess the performance of the weights is the area under the curve (AUC) of the ROC curve of the classifier trained to differentiate the target sample from the nominal sample and from the nominal sample *with* the weights applied, as shown below. 

If the classifier is able to discriminate between the two samples the resulting AUC is larger than 0.5, as in the case of comparing the original sample `p0(x)` and the target sample `p1(x)` (blue curve).  On the other hand, if the weights are applied to the nominal sample, the classifier *is unable to discriminate* the target sample from the weighted original sample (orange curve), which results in an AUC of exactly 0.5, meaning that the weighted original sample and the target one are *virtually identical!*.    
<p align="center">
<img src="https://github.com/leonoravesterbacka/carl-torch/blob/master/images/roc_nominalVsQSFDOWN_dilepton_train_True.png" width="400">
</p>
The foreseen gain using this method is the ability to mimic a variational sample by generating weights using much smaller samples (down to 1/20th of the number of events of the nominal sample), instead of generating each variational sample with the same statistical power as the nominal one. As the training of the classifier that is the base of the weight derivation is much lighter in terms of CPU than the full Monte-Carlo sample generation, this method has immense potential in reducing CPU usage. 

## Documentation
Extensive details regarding likelihood-free inference with calibrated classifiers can be found in the paper _"Approximating Likelihood Ratios with Calibrated Discriminative Classifiers", Kyle Cranmer, Juan Pavez, Gilles Louppe._ [http://arxiv.org/abs/1506.02169](http://arxiv.org/abs/1506.02169), as well as Kyles keynote at [NeurIPS 2016 (slide 57)](https://figshare.com/articles/journal_contribution/NIPS_2016_Keynote_Machine_Learning_Likelihood_Free_Inference_in_Particle_Physics/4291565/1).
## Installation
The following dependencies are required:
 - numpy>=1.13.0
 - scipy>=1.0.0
 - scikit-learn>=0.19.0
 - torch>=1.0.0
 - uproot
 - matplotlib>=2.0.0
 - onnxruntime>=1.5.0

For hyperparameter optimization skorch is also required. 

Once satisfied, `carl-torch` can be installed from source using the following:
```
git clone https://github.com/leonoravesterbacka/carl-torch.git
```

## Execution
The code is based on three scripts:
- [train.py](train.py) trains neural networks to discriminate between two simulated samples.
- [evaluate.py](evaluate.py) evaluates the neural network by calculating the weights and making validation and ROC plots.
- [calibrate.py](calibrate.py) calibrated network predictions based on histograms of the network output.

Validation plots are made with option plot set to True in [evaluate.py](evaluate.py), and saved in plots/. 

Hyperparameter search for optimization of the classifier is done in branch hyperparameter-search using skorch.

The training is preferrably done on GPUs. [HTCondor_README.md](HTCondor_README.md) includes instructions on how to train on GPUs on HTCondor (ATLAS users only for now). The evaluation and calibration steps are done instantly and thus not require GPUs. 

## Deployment
The model trained in the train.py step is exported to [onnx](https://github.com/onnx/onnx) format to be loaded in a C++ production environment using [onnxruntime](https://github.com/microsoft/onnxruntime). 
#### For ATLAS users
The [carlAthenaOnnx](https://gitlab.cern.ch/mvesterb/carlathenaonnx/-/tree/master/carlAthenaOnnx) is a package that loads the models trained with carl-torch in AthDerivation production environment, with the purpose of centrally providing the weights for each theory variation to the user.  
In order to validate that the weights infered using carl-torch agree with weights infered through an external deployment, the validate.py script can be used:
- [validate.py](validate.py) compares the weights obtained from carl-torch agrees with weights infered through external deployment procedure, such as using [carlAthenaOnnx](https://gitlab.cern.ch/mvesterb/carlathenaonnx/-/tree/master/carlAthenaOnnx). N.B. carlAthenaOnnx now supports weight derivation for both EVNT and xAOD samples. 
## Support
If you have any questions, please email leonora.vesterbacka@cern.ch
