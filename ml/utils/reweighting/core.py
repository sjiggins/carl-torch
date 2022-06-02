from . import core

import json


def binned_reweighting(x0, w0, x1, w1, config_file):
    """
    Args:
        x0 : pandas.DataFrame
            dataframe contains feature from sample-0

        w0 : pandas.DataFrame
            dataframe contains event weight from sample-0

        x1 : pandas.DataFrame
            dataframe contains feature from sample-1

        w1 : pandas.DataFrame
            dataframe contains event weight from sample-1

        config_file : str
            configuration file with the following format
            {
                "method" : "binned_1D_reweighting"
                "arguments" : {
                    "bins" : [1,2,3,4],
                    "observable" : "Njets"
                    "normalise" : True
                }
            }
    """

    with open(config_file, "r") as f:
        config_file = json.load(f)

    reweight_method = getattr(core, config_file["method"])
    args = getattr(core, config_file["arguments"])

    return reweight_method(x0, w0, x1, w1, **args)
