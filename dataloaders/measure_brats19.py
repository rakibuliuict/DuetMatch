import os
import nibabel as nib
import numpy as np
from collections import Counter

INPUT_ROOT = r"D:\Dataset\BraTS_2019\BraTS_Flair_seg\BraTS_2019"  

def get_brain_bbox(flair_data, seg_data, margin=0):
    mask = (flair_data != 0) | (seg_data != 0)
    if not np.any(mask):
        return None

    coords = np.where(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    z_min = max(z_min - margin, 0)

    x_max = min(x_max + margin, flair_data.shape[0] - 1)
    y_max = min(y_max + margin, flair_data.shape[1] - 1)
    z_max = min(z_max + margin, flair_data.shape[2] - 1)

    return (slice(x_min, x_max + 1),
            slice(y_min, y_max + 1),
            slice(z_min, z_max + 1))

def main():
    shapes = []

    for root, dirs, files in os.walk(INPUT_ROOT):
        has_nifti = any(
            f.endswith("_flair.nii") or f.endswith("_flair.nii.gz") or
            f.endswith("_seg.nii") or f.endswith("_seg.nii.gz")
            for f in files
        )
        if not has_nifti:
            continue

        flair_file = None
        seg_file = None
        for f in files:
            if f.endswith("_flair.nii") or f.endswith("_flair.nii.gz"):
                flair_file = f
            elif f.endswith("_seg.nii") or f.endswith("_seg.nii.gz"):
                seg_file = f

        if flair_file is None or seg_file is None:
            continue

        flair_path = os.path.join(root, flair_file)
        seg_path = os.path.join(root, seg_file)

        flair_img = nib.load(flair_path)
        seg_img = nib.load(seg_path)
        flair_data = flair_img.get_fdata()
        seg_data = seg_img.get_fdata()

        bbox = get_brain_bbox(flair_data, seg_data)
        if bbox is None:
            continue

        xs, ys, zs = bbox
        cropped_shape = flair_data[xs, ys, zs].shape
        shapes.append(cropped_shape)

        print(f"{os.path.relpath(root, INPUT_ROOT)} -> {cropped_shape}")

    shapes = np.array(shapes)
    print("\n=== DATASET CROP STATISTICS ===")
    print("Num cases:", len(shapes))
    print("Min shape (Dx, Dy, Dz):", shapes.min(axis=0))
    print("Max shape (Dx, Dy, Dz):", shapes.max(axis=0))

    # simple frequency info
    cnt = Counter([tuple(s) for s in shapes])
    print("\nMost common shapes:")
    for shape, c in cnt.most_common(10):
        print(f"{shape}: {c} cases")

    # suggest padded shape (round up to multiple of 16)
    max_shape = shapes.max(axis=0)
    suggested = tuple(int(np.ceil(d / 16.0) * 16) for d in max_shape)
    print("\nSuggested padded shape (multiple of 16):", suggested)


if __name__ == "__main__":
    main()
