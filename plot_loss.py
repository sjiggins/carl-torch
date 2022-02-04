import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="usage: %prog [opts]")
    parser.add_argument('--version', action='version', version='%prog 1.0')
    parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
    parser.add_argument('-d', '--dir', action="store", type=str, dest="dir", help="Directory to the data files.")
    opts = parser.parse_args()

    stats_methods = [
        "compute_kl_divergence",
        "wasserstein",
        #"chisquare",
    ]

    # plotting loss vs epoch
    for type in ["accuracy", "loss"]:
        train_loss = f"{opts.dir}/{type}_train_{opts.global_name}.npy"
        val_loss = f"{opts.dir}/{type}_val_{opts.global_name}.npy"
        if all(map(os.path.exists, [train_loss, val_loss])):
            train_loss = np.load(train_loss)
            val_loss = np.load(val_loss)
            fig, ax1 = plt.subplots(1,1, figsize=(12,10))

            ax1.set_xlabel("Epoch", loc="right", fontsize=18)
            ax1.plot(train_loss, "b-", label=f"Train {type}")
            ax1.set_ylabel(f"Train {type}", fontsize=18)
            ax1.tick_params(axis='both', labelsize=18)

            ax2 = ax1.twinx()
            ax2.plot(val_loss, "r-", label=f"Val {type}")
            ax2.set_ylabel(f"Val {type}", fontsize=18)
            ax2.tick_params(axis='both', labelsize=18)

            line1, label1 = ax1.get_legend_handles_labels()
            line2, label2 = ax2.get_legend_handles_labels()
            lines = line1 + line2
            labels = label1 + label2
            ax1.legend(
                lines,
                labels,
                loc=0,
                frameon=False,
                title=f"{type} vs Epoch",
                fontsize=18,
                title_fontsize=18,
            )
            fig.savefig(f"{opts.dir}/{type}_vs_epoch.png")

    for method in stats_methods:
        for type in ["train", "val"]:
            ifile_format =  f"{opts.dir}/stats_dist/{type}*{method}.npy"
            fig, ax1 = plt.subplots(1,1, figsize=(12,10))
            ax1.set_xlabel("Epoch", loc="right", fontsize=18)
            ax1.set_ylabel(f"value of {method}", fontsize=18)
            ax1.tick_params(axis='both', labelsize=18)
            for ifile in glob.glob(ifile_format):
                obs_name = os.path.basename(ifile).replace(f"{type}_", "")
                obs_name = obs_name.replace(f"_{method}.npy", "")
                data = np.load(ifile)
                ax1.plot(data, label=f"Obs {obs_name}")
            ax1.legend(frameon=False, fontsize=18)
            fig.savefig(f"{opts.dir}/{type}_{method}_vs_epoch.png")
