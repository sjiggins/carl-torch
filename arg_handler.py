#import optparser
import argparse

#################################################
# Argument parsing helper functions
def percentile_range(value):
    value = float(value)
    # Restrict percentile to a range of 0-100
    if value > 100. or value < 0.:
        raise argparse.ArgumentTypeError("%s should be between 0-100, with float precision allowed." % value)
    return value
#################################################

# Argument parsing main function for training step
def arg_handler_train():
    parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
    parser.add_argument('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
    parser.add_argument('-e', '--nentries',  action='store', type=int, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
    parser.add_argument('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
    parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
    parser.add_argument('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
    parser.add_argument('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='DummyEvtWeight', help='Name of event weights feature in TTree')
    parser.add_argument('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
    parser.add_argument('-b', '--binning',  action='store', type=str, dest='binning',  default=None, help='path to binning yaml file.')
    parser.add_argument('-l', '--layers', action='store', type=int, dest='layers', nargs='*', default=None, help='number of nodes for each layer')
    parser.add_argument('-d', '--dropout-prob', action='store', type=float, dest='dropout_prob', default=None, help='Dropout probability for internal hidden layers')
    parser.add_argument('-r', '--regularise', action='store', type=str, dest='regularise', default=None, help='Regularisation technique for the loss function [L0, L1, L2]')
    parser.add_argument('-a', '--algorithms', action='store', type=str, dest='algorithms', default=None, help='List of algorithms to run for data processing. Options = ["subsample"]')
    parser.add_argument('--batch',  action='store', type=int, dest='batch_size',  default=4096, help='batch size')
    parser.add_argument('--per-epoch-plot', action='store_true', dest='per_epoch_plot', default=False, help='plotting train/validation result per epoch.')
    parser.add_argument('--per-epoch-save', action='store_true', dest='per_epoch_save', default=False, help='saving trained model per epoch.')
    parser.add_argument('--nepoch', action='store', dest='nepoch', type=int, default=300, help='Total number of epoch for training.')
    parser.add_argument('--scale-method', action='store', dest='scale_method', type=str, default="minmax", help='scaling method for input data. e.g minmax, standard.')
    parser.add_argument('--weight-clipping', action='store_true', dest='weight_clipping', default=False, help='clipping event weights')
    parser.add_argument('--weight-nsigma', action='store', type=int, dest='weight_nsigma', default=0, help='re-mapping weights')
    parser.add_argument('--polarity', action='store_true', dest="polarity", help='enable event weight polarity feature.')
    parser.add_argument('--loss-type', action='store', type=str, dest="loss_type", default="regular", help='a type on how to handle weight in loss function, options are "abs(w)" & "log(abs(w))" ')
    parser.add_argument('--BoolFilter', action='store', dest='BoolFilter', type=str, default=None, help='Comma separated list of boolean logic. e.g. \'a | b\'.')
    opts = parser.parse_args()
    return opts

# Arugment parsing main function for evaluation step
def arg_handler_eval():
    parser = argparse.ArgumentParser(usage="usage: %prog [opts]")
    parser.add_argument('-n', '--nominal',   action='store', type=str, dest='nominal',   default='', help='Nominal sample name (root file name excluding the .root extension)')
    parser.add_argument('-v', '--variation', action='store', type=str, dest='variation', default='', help='Variation sample name (root file name excluding the .root extension)')
    parser.add_argument('-e', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events to do the training on, None means full sample')
    parser.add_argument('-p', '--datapath',  action='store', type=str, dest='datapath',  default='./Inputs/', help='path to where the data is stored')
    parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
    parser.add_argument('-f', '--features',  action='store', type=str, dest='features',  default='', help='Comma separated list of features within tree')
    parser.add_argument('-w', '--weightFeature',  action='store', type=str, dest='weightFeature',  default='', help='Name of event weights feature in TTree')
    parser.add_argument('-t', '--TreeName',  action='store', type=str, dest='treename',  default='Tree', help='Name of TTree name inside root files')
    parser.add_argument('--PlotROC',  action="store_true", dest='plot_ROC',  help='Flag to determine if one should plot ROC')
    parser.add_argument('--PlotObsROC',  action="store_true", dest='plot_obs_ROC',  help='Flag to determine if one should plot observable ROCs')
    parser.add_argument('--PlotResampleRatio',  action="store_true", dest='plot_resampledRatio',  help='Flag to determine if one should plot a ratio of resampled vs original distribution')
    parser.add_argument('-m', '--model', action='store', type=str, dest='model', default=None, help='path to the model')
    parser.add_argument('-b', '--binning',  action='store', type=str, dest='binning',  default=None, help='path to binning yaml file')
    parser.add_argument('--normalise', action='store_true', dest='normalise', default=False, help='enforce normalization when plotting')
    parser.add_argument('--rawWeight',  action="store_true", dest='raw_weight',  help='Flag to use raw event weight')
    parser.add_argument('--scale-method', action='store', dest='scale_method', type=str, default=None, help='scaling method for input data. e.g minmax, standard')
    parser.add_argument('--weight-protection', action='store_true', dest='weight_protection',  default=False, help='implement CARL weight protection by clipping away (+-) inf values')
    parser.add_argument('--weight-threshold', action='store', dest='weight_threshold', type=percentile_range, default=100, help='implement CARL weight clipping by clipping based on user defined percentile. Valid range from [0,100]% with floating precision')
    opts = parser.parse_args()
    return opts
