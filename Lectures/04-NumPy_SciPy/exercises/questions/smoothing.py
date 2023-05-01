import numpy as np
from scipy import ndimage


def smooth(img, sigma=1):
    """Create a copy of image smoothed with a Gaussian filter."""
    # TODO
    # Hint: look for scipy.ndimage.gaussian_filter


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images = np.load("images.npz")
    img = images["mni_template"]
    smoothed_img = smooth(img)

    plt.imshow(img, cmap="gray")
    plt.figure()
    plt.imshow(smoothed_img, cmap="gray")
    plt.show()
