import matplotlib.pyplot as plt
import numpy as np


def hist_feature(x, hist_png):
    x = x.detach().cpu().numpy().flatten()
    # q25, q75 = np.percentile(x, [0.25, 0.75])
    # bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    # bins = round((x.max() - x.min()) / bin_width)
    # print("Freedman–Diaconis number of bins:", bins)
    print(x.max(), x.min())
    plt.hist(x, bins=100)
    plt.savefig(hist_png)
