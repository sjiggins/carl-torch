import os
import json
import logging

logger = logging.getLogger(__name__)


def datafiles_path_preparation(
    data_path, tag_name, nevent, user_config=None, skip_ok=True
):
    """
    Prepare data files path for used in the evaluation/calibration.

    Args:
        data_path : str
            path to the training directory.

        tag_name : str
            tag name used in the training.

        nevents: int
            number of events used in the training.

        user_config: str
            path to JSON file used for overwriting default file path.

        skip_ok: bool, default=True
            raise exception if one of the files doesn't not exist.
    """

    metadata = f"{data_path}/data/{tag_name}/metaData_{nevent}.pkl"
    spec_meta = f"{data_path}/data/{tag_name}/metaData_{nevent}_spectator.pkl"

    # should hamonize the file naming in future
    features_file_map = {"x0": "X0", "x1": "X1", "w0": "w0", "w1": "w1"}
    spectator_file_map = {"x0": "spec_x0", "x1": "spec_x1", "w0": "w0", "w1": "w1"}

    # setting up default data files
    data_files_path = {
        "model": f"{data_path}/models/{tag_name}_carl_{nevent}",
        "train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in features_file_map.items()
        },
        "val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in features_file_map.items()
        },
        "spectator-train": {
            x: f"{data_path}/data/{tag_name}/{y}_train_{nevent}.npy"
            for x, y in spectator_file_map.items()
        },
        "spectator-val": {
            x: f"{data_path}/data/{tag_name}/{y}_val_{nevent}.npy"
            for x, y in spectator_file_map.items()
        },
    }
    # add meta data path
    for key in ["train", "val"]:
        data_files_path[key].update({"metaData": metadata})
    for key in ["spectator-train", "spectator-val"]:
        data_files_path[key].update({"metaData": spec_meta})

    # check user specified data files
    if user_config is not None:
        with open(user_config, "r") as f:
            user_fdata = json.load(f)
            logger.info(f"user file data: {user_fdata}")
            data_files_path.update(user_fdata)
    logger.info(f"data file: {data_files_path}")

    # check trained files
    check_lists = {
        "train": data_files_path["train"],
        "val": data_files_path["val"],
        "spec-train": data_files_path["spectator-train"],
        "spec-val": data_files_path["spectator-val"],
    }
    for name, check_list in check_lists.items():
        logger.info(f"Checking files for {name}")
        flist = check_list.values()
        if all(map(os.path.exists, flist)):
            continue
        _err_msg = f"Cannot locate all of the files {flist}"
        if not skip_ok:
            raise FileNotFoundError(_err_msg)
        logger.warning(_err_msg)
        # setting it to empty dict
        check_list = {}

    return data_files_path
