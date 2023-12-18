"""Putting it all together -- using a brain atlas.

Here we apply all the necessary transformations to the original timeseries
images so we can perform numerical analysis on them, and eventually map the
result back into a brain image.

The steps are:

- smooth the input images
- resample the atlas to the images' resolution
- transform the images (average timeseries within atlas regions).
"""
import numpy as np
from matplotlib import pyplot as plt

# Note we import the modules we created in the previous exercises
import atlas_transformations
import resampling
import smoothing

images = np.load("images.npz")

img = images["difumo_misaligned"]
img_affine = images["difumo_misaligned_affine"]

harvard_oxford_atlas = images["harvard_oxford_atlas"]
harvard_oxford_atlas_affine = images["harvard_oxford_atlas_affine"]

# TODO: resample the atlas to the difumo images' resolution
# atlas_affine, atlas = ...

# TODO: smooth the input images
# smoothed_img = ...

# TODO: transform the timeseries
# transformed_data = ...
print(f"transformed data shape: {transformed_data.shape}")

# Here we could perform our analyses on our 2D array of (region, time)
# timeseries.

# TODO: apply the inverse transform to the timeseries to transform them back to
# an image
# new_img = ...
print(f"unmasked image shape: {new_img.shape}")

# plotting the original and transformed images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img[..., 3])
axes[1].imshow(new_img[..., 3], interpolation="nearest")

plt.show()
