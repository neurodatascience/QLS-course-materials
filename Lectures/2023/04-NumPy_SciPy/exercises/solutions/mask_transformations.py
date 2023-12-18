import numpy as np


def mask_transform(img, mask_img):
    """Extract timeseries corresponding to pixels inside a mask.

    Parameters
    ----------
    img: 3D numpy array (2D images x time). The data to mask.

    mask_img: 2D numpy array. An image of the same shape as `img[..., 0]`,
    contains `True` for pixels that should be kept and `False` for voxels to be
    discarded.

    Returns
    -------
    A 2D array of shape (N kept pixels, N time points). The extracted
    timeseries, corresponding to non-zero entries in `mask_img`, ordered in C
    (row-major) order.

    """
    assert np.ndim(img) == 3
    # TODO
    # TODO_BEGIN
    return img[mask_img]


# TODO_END


def mask_inverse_transform(data, mask_img):
    """Inverse transform of `mask_transform`.

    Parameters:
    -----------
    data: 2D numpy array (pixel index x time) of time series.

    mask_img: mask used to produce `data`, the number of nonzero entries should
    match the first dimension of `data`.

    Returns:
    --------
    A 3D array containing the unmasked data. The first 2 dimensions have the
    same shape as `mask_img` and the last has the same size as the second
    dimension of `data`.

    """
    # TODO
    # TODO_BEGIN
    img = np.zeros((*mask_img.shape, data.shape[1]))
    img[mask_img] = data
    return img


# TODO_END


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images = np.load("images.npz")
    mni_mask = images["mni_mask"]
    img = images["difumo"]

    masked_data = mask_transform(img, mni_mask)
    print(f"masked data shape: {masked_data.shape}")

    unmasked = mask_inverse_transform(masked_data, mni_mask)
    print(f"unmasked image shape: {unmasked.shape}")

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[..., 3], cmap="seismic", vmin=-1, vmax=1)
    axes[1].imshow(unmasked[..., 3], cmap="seismic", vmin=-1, vmax=1)

    plt.show()
