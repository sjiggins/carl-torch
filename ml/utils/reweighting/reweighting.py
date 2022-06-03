"""
Module for handling weight reweighting.
"""
import numpy as np

# ===============================================================================
# ===============================================================================


def generate_1d_histogram(name, bins, df0, df1, w0, w1, *args, **kwargs):
    m_bins = np.asarray(bins)
    m_bins = np.concatenate([np.array([-np.inf]), m_bins, np.array([np.inf])])
    hist0 = np.histogram(df0[name], weights=w0, bins=m_bins, *args, **kwargs)
    hist1 = np.histogram(df1[name], weights=w1, bins=m_bins, *args, **kwargs)
    return hist0, hist1


# ===============================================================================
# ===============================================================================
def binned_1D_reweighting(
    df_feature0,
    df_w0,
    df_feature1,
    df_w1,
    observable,
    bins,
    normalise=False,
    *args,
    **kwargs,
):
    """
    This function provides a binned reweighting method for event weights of two
    sample sets. For example, using a (binning) jet multiplicity histogram and
    rescaling the event weight per bin. At the end, the sample-0 (nominal) will
    have unity values per bin the jet multiplicity. The same scale used for rescaling
    is applied to the sample-1 (variation) to maintin the relative difference.
    Only support 1D binned re-weighting for now.

    Args:
        df_feature0 : pandas.DataFrame
            dataframe contains feature from sample-0

        df_w0 : pandas.DataFrame
            dataframe contains event weight from sample-0

        df_feature1 : pandas.DataFrame
            dataframe contains feature from sample-1

        df_w1 : pandas.DataFrame
            dataframe contains event weight from sample-1

        observable : str
            name of the feature use for bin reweighting

        bins : list
            histogram bin edges.
    """
    if normalise:
        df_w0 = df_w0.divide(df_w0.iloc[:, 0].sum(), axis=0)
        df_w1 = df_w1.divide(df_w1.iloc[:, 0].sum(), axis=0)

    hist0, hist1 = generate_1d_histogram(
        observable,
        bins,
        df_feature0,
        df_feature1,
        w0=df_w0.iloc[:, 0],
        w1=df_w1.iloc[:, 0],
        *args,
        **kwargs,
    )

    hist0, hist0_edge = hist0
    hist1, hist1_edge = hist1

    # nomalise bin content
    hist0 /= np.sum(hist0)
    hist1 /= np.sum(hist1)

    # getting the bin index
    # digitizing with the same bin edges
    bin_index0 = np.digitize(df_feature0[observable], hist0_edge)
    bin_index1 = np.digitize(df_feature1[observable], hist0_edge)

    # need to grab the content from the
    # same sample so the scaling is the same for both samples.
    bin_content0 = hist0[bin_index0 - 1]  # shift by overflow bin
    # bin_content1 = hist0[bin_index1 - 1]
    bin_content1 = hist1[bin_index1 - 1]

    # just find the ratio
    # hist_ratio = hist0 / hist1
    # reset all nominal weight to 1
    # df_w0.iloc[:, 0] = 1.0
    # scale the ratio
    # df_w1.iloc[:, 0] = 1.0 / hist_ratio[bin_index1 - 1]

    # multiply by the bin content to increase the weighting
    # df_w0 = df_w0.multiply(bin_content0, axis=0)  # can we do in-place?
    # df_w1 = df_w1.multiply(bin_content1, axis=0)

    df_w0.iloc[:, 0] = bin_content0
    df_w1.iloc[:, 0] = bin_content1

    return df_feature0, df_w0, df_feature1, df_w1
