"""Putting it all together -- using a brain mask.

Here we apply all the necessary transformations to the original timeseries
images so we can perform numerical analysis on them, and eventually map the
result back into a brain image.

The steps are:

- smooth the input images
- resample the mask to the images' resolution
- mask the images
"""
import numpy as np
from matplotlib import pyplot as plt

# Note we import the modules we created in the previous exercises
import mask_transformations
import resampling
import smoothing

images = np.load("images.npz")

img = images["difumo_misaligned"]
img_affine = images["difumo_misaligned_affine"]

mni_mask = images["mni_mask"]
mni_mask_affine = images["mni_mask_affine"]

# TODO: resample the mask to the difumo images' resolution
# mask_affine, mask = ...

# TODO: smooth the input images
# smoothed_img = ...

# TODO: mask the timeseries
# masked_data = ...
print(f"masked data shape: {masked_data.shape}")

# Here we could perform our analyses on our 2D array of (pixel, time)
# timeseries.

# TODO: unmask the timeseries to transform them back to an image
# unmasked = ...

# plotting the original and unmasked images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img[..., 3], cmap="seismic", vmin=-img.max(), vmax=img.max())
axes[1].imshow(
    unmasked[..., 3],
    cmap="seismic",
    vmin=-unmasked.max(),
    vmax=unmasked.max(),
)

plt.show()
