from . import reweighting

import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    logger.info(f"reading in reweighting setting from {config_file}")

    with open(config_file, "r") as f:
        config_file = json.load(f)

    method_name = config_file["method"]
    args = config_file["arguments"]
    logger.info(f"using rewighting method {method_name}")
    reweight_method = getattr(reweighting, method_name)

    return reweight_method(x0, w0, x1, w1, **args)
