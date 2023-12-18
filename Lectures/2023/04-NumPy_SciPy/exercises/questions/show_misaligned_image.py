import numpy as np
from matplotlib import pyplot as plt

images = np.load("images.npz")

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

axes[2].imshow(images["difumo_misaligned"].max(axis=-1))
axes[2].set_title("original timeseries that we will be using from now on" "\n(max along time axis)")

axes[0].imshow(images["mni_template"], cmap="gray")
axes[0].set_title("MNI template")

axes[1].imshow(images["difumo"].max(axis=-1))
axes[1].set_title(
    "timeseries that have been resampled to the MNI template" "\n(max along time axis)"
)

plt.show()
