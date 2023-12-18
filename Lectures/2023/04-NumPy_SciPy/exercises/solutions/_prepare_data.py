import numpy as np
from matplotlib import pyplot as plt
from nilearn import datasets, image

mni_mask = datasets.load_mni152_brain_mask(1)
mni_template = datasets.load_mni152_template(1)
harvard_oxford = image.resample_to_img(
    datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")["maps"],
    mni_template,
    interpolation="nearest",
)


def make_difumo():
    difumo = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)["maps"]
    difumo_data = image.get_data(difumo)
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=difumo_data.max() / 16, size=difumo_data.shape).astype("float32")
    difumo_mask = image.get_data(
        image.resample_to_img(datasets.load_mni152_brain_mask(2), difumo, interpolation="nearest")
    )
    noise += difumo_mask[..., None] * rng.normal(
        scale=difumo_data.max() / 8, size=difumo_data.shape
    ).astype("float32")
    difumo_data += noise
    difumo_data /= difumo_data.max()
    difumo = image.new_img_like(difumo, difumo_data)
    return difumo


difumo_misaligned = make_difumo()
difumo = image.resample_to_img(
    difumo_misaligned,
    mni_template,
    interpolation="nearest",
)
difumo_misaligned = image.resample_img(
    difumo_misaligned,
    target_affine=np.asarray([[0, 2.0, 0], [-2.0, 0, 0], [0, 0, 2.0]]),
    interpolation="nearest",
)

assert (mni_template.affine == mni_mask.affine).all()
assert (harvard_oxford.affine == mni_mask.affine).all()
assert (difumo.affine == mni_mask.affine).all()

assert mni_template.shape == mni_mask.shape
assert harvard_oxford.shape == mni_mask.shape
assert difumo.shape == (*mni_mask.shape, 64)


z_slice = 100
mni_template_data = image.get_data(mni_template)[..., z_slice]
mni_mask_data = image.get_data(mni_mask)[..., z_slice]
harvard_oxford_data = image.get_data(harvard_oxford)[..., z_slice]
regions = np.unique(harvard_oxford_data)
rindex = np.zeros(harvard_oxford_data.max() + 1)
rindex[regions] = np.arange(len(regions))
harvard_oxford_data = rindex[harvard_oxford_data].astype("int8")
difumo_data = image.get_data(difumo)[..., z_slice, :]
difumo_misaligned_data = image.get_data(difumo_misaligned)[..., z_slice // 2, :]


assert mni_mask_data.dtype == "int8"
mni_mask_data = mni_mask_data.astype("bool")
assert mni_template_data.dtype == "float32"
assert harvard_oxford_data.dtype == "int8"
assert difumo_misaligned_data.dtype == "float32"
assert difumo_data.dtype == "float32"


def slice_affine(affine):
    idx = [0, 1, 3]
    return affine[np.ix_(idx, idx)]


np.savez_compressed(
    "images.npz",
    mni_mask=mni_mask_data,
    mni_mask_affine=slice_affine(mni_mask.affine),
    mni_template=mni_template_data,
    mni_template_affine=slice_affine(mni_template.affine),
    harvard_oxford_atlas=harvard_oxford_data,
    harvard_oxford_atlas_affine=slice_affine(harvard_oxford.affine),
    difumo_misaligned=difumo_misaligned_data,
    difumo_misaligned_affine=slice_affine(difumo_misaligned.affine),
    difumo=difumo_data,
    difumo_affine=slice_affine(difumo.affine),
)

images = np.load("images.npz")
mni_mask_data = images["mni_mask"]
mni_template_data = images["mni_template"]
harvard_oxford_data = images["harvard_oxford_atlas"]
difumo_data = images["difumo"]
difumo_misaligned_data = images["difumo_misaligned"]

fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)

axes[4].imshow(difumo_misaligned_data.max(axis=-1), cmap="bwr")
axes[0].imshow(mni_mask_data, cmap="gray")
axes[1].imshow(mni_template_data, cmap="gray")
axes[2].imshow(harvard_oxford_data, cmap="gist_ncar")
axes[3].imshow(difumo_data.max(axis=-1), cmap="bwr")

plt.show()
