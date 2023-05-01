import numpy as np
from scipy import ndimage


def resample(source_spatial_img, target_spatial_img):
    """Resample a source image so it is aligned with the target image.

    Parameters
    ----------

    source_spatial_img: Tuple (source_affine, source_img). The source_affine is
    the affine transformation that maps pixel indices in source_img to world
    coordinates. `source_img` is a 2D image, a numpy array with 2 dimensions.
    Therefore `source_affine` has shape (3, 3).

    target_spatial_img: Tuple (target_affine, target_img). The target_affine is
    the affine transformation that maps pixel indices in target_img to world
    coordinates. `target_img` is a 2D image or a timeseries of 2D images, ie a
    numpy array with 2 or 3 dimensions.
    Therefore `target_affine` has shape (3, 3).

    Returns
    -------

    target_affine, transformed_img: `target_affine` is the same as the
    corresponding argument. `transformed_img` is the `source_img` resampled to
    be aligned with `target_img`. It has the same shape in the first 2
    dimensions as `target_img`.

    """
    source_affine, source_img = source_spatial_img
    assert source_img.ndim == 2
    target_affine, target_img = target_spatial_img
    # TODO:
    # compute the affine transformation from the target to the source
    # coordinate system. We will later use scipy.ndimage.affine_transform,
    # which needs, for (i, j) pixel indices in the new (resampled) image, the
    # corresponding index in the original image.
    #
    # Therefore we first map (i, j) pixel index to world coordinates by
    # applying the `target_affine`, then map the resulting world coordinates to
    # source indices (i', j') by applying the inverse of the source affine.
    #
    # You can use `numpy.linalg.inv` to invert an inversible matrix.
# TODO_BEGIN
    transform = np.linalg.inv(source_affine) @ target_affine
# TODO_END

    # The last row of the augmented matrix representing the transformation is
    # always [0, 0, 1]. This will be already the case but up to small numerical
    # errors due to the inversion, so we set it to be exactly [0, 0, 1] here.
    transform[-1] = [0.0, 0.0, 1.0]

    # TODO:

    # Apply the transformation. Use `scipy.ndimage.affine_transform`. As we
    # will be resampling binary masks, we don't want continuous interpolation
    # so set `order=0`.
# TODO_BEGIN
    transformed_img = ndimage.affine_transform(
        source_img, transform, order=0, output_shape=target_img.shape[:2]
    )
# TODO_END
    return target_affine, transformed_img


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images = np.load("images.npz")
    mni_mask = images["mni_mask"]
    img = images["difumo_misaligned"]
    img_affine = images["difumo_misaligned_affine"]
    template = images["mni_template"]
    template_affine = images["mni_template_affine"]

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    resampled_template_affine, resampled_template = resample(
        (template_affine, template), (img_affine, img)
    )
    axes[0].imshow(template)
    axes[1].imshow(resampled_template)
    axes[2].imshow(img.max(axis=-1))
    plt.show()
