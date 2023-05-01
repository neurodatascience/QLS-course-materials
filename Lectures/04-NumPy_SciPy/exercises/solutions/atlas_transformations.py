import numpy as np


def atlas_transform(img, atlas_img):
    """Extract averaged timeseries corresponding to regions in an atlas.

    Parameters
    ----------
    img: 3D numpy array (2D images x time). The data to transform.

    atlas_img: 2D numpy array. An image of the same shape as `img[..., 0]`,
    contains integers indicating the region ID of each pixel. All pixels in a
    region will be averaged to produce 1 average timeseries for the region.
    Region IDs must range from 0 to N regions (this is the case for the
    `harvard_oxford_atlas` atlas provided for this exercise).

    Returns
    -------
    A 2D array of shape (N regions, N time points). The extracted timeseries,
    corresponding to regions in `atlas_img`, ordered by region ID.

    """
    assert np.ndim(img) == 3
    assert (np.unique(atlas_img) == np.arange(atlas_img.max() + 1)).all()

    # TODO
    # Hint: look for the documentation for the `numpy.bincount` function. It
    # can count voxels in the different regions. Moreover it accepts a
    # `weights` parameter that allows summing arbitrary values grouped by
    # region (rather than the default weight, 1, which produces the counts).
# TODO_BEGIN
    flat_atlas_img = atlas_img.ravel()
    region_counts = np.bincount(flat_atlas_img)
    result = np.empty((len(region_counts), img.shape[-1]))
    for i in range(img.shape[-1]):
        region_sums = np.bincount(flat_atlas_img, weights=img[..., i].ravel())
        result[:, i] = region_sums / region_counts
    return result
# TODO_END


def atlas_inverse_transform(data, atlas_img):
    """Inverse transform of `atlas_transform`.

    Parameters:
    -----------
    data: 2D numpy array (region ID x time) of time series.

    atlas_img: atlas used to produce `data`, the number of regions should match
    the first dimension of `data`.

    Returns:
    --------
    A 3D array containing the unmasked data. The first 2 dimensions have the
    same shape as `atlas_img` and the last has the same size as the second
    dimension of `data`.

    """
    # TODO
    # Hint: the timeseries in `data` are ordered by region ID. You can use
    # advanced integer indexing...
# TODO_BEGIN
    return data[atlas_img]
# TODO_END


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images = np.load("images.npz")
    harvard_oxford = images["harvard_oxford_atlas"]
    img = images["difumo"]

    data = atlas_transform(img, harvard_oxford)
    print(f"masked data shape: {data.shape}")

    new_img = atlas_inverse_transform(data, harvard_oxford)
    print(f"unmasked image shape: {new_img.shape}")

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[..., 3])
    axes[1].imshow(new_img[..., 3], interpolation="nearest")

    plt.show()
