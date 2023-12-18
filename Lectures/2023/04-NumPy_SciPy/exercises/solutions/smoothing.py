import numpy as np
from scipy import ndimage


def smooth(img, sigma=2):
    """Create a copy of image smoothed with a Gaussian filter."""
    # TODO
    # Hint: look for scipy.ndimage.gaussian_filter
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    #
    # Hint: if img is 3D, we don't want to smooth along the last dimension...
    # `gaussian_filter` allows specifying a different `sigma` for each
    # dimension.
    # TODO_BEGIN
    full_sigma = np.ones(img.ndim) * sigma
    full_sigma[2:] = 0.0
    return ndimage.gaussian_filter(img, full_sigma)


# TODO_END


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images = np.load("images.npz")
    template = images["mni_template"]
    smoothed_template = smooth(template)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(template, cmap="gray")
    axes[1, 0].imshow(smoothed_template, cmap="gray")

    img = images["difumo"]
    axes[0, 1].imshow(img[..., 3])
    smoothed_img = smooth(img)
    axes[1, 1].imshow(smoothed_img[..., 3])
    for i in range(5):
        assert np.allclose(smoothed_img[..., i], smooth(img[..., i]))

    plt.show()
