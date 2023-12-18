import numpy as np
from matplotlib import pyplot as plt

images = np.load("images.npz")

comments = {
    "mni_template": "MNI anatomical template",
    "mni_mask": "MNI brain mask",
    "harvard_oxford_atlas": "Harvard-Oxford atlas",
}
fig, axes = plt.subplots(1, 3)
axes[0].imshow(images["mni_template"], cmap="gray")
axes[1].imshow(images["mni_mask"], cmap="gray")
axes[2].imshow(
    images["harvard_oxford_atlas"],
    cmap="nipy_spectral",
    interpolation="nearest",
)
for ax, (key, comment) in zip(axes, comments.items()):
    ax.set_title(f"{comment}\nimages['{key}']: {images[key].dtype}")

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("First 16 'time points' of 'difumo'")
for ax, img_slice in zip(axes.ravel(), np.moveaxis(images["difumo"], -1, 0)):
    ax.imshow(img_slice, cmap="inferno")


fig, ax = plt.subplots()
fig.suptitle("Example 'timeseries' from 'difumo'")
ax.plot(images["difumo"][100:105, 100:105, :].reshape(-1, images["difumo"].shape[-1]).T)

plt.show()
