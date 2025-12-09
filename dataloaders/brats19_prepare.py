import os
import nibabel as nib
import numpy as np

# ========= CONFIG =========
INPUT_ROOT = r"D:\Dataset\BraTS_2019\BraTS_Flair_seg\BraTS_2019"                 # original data root
OUTPUT_ROOT = r"D:\Dataset\BraTS_2019\Crop_Flair"         # output root
TARGET_SHAPE = (192, 192, 160)  # (Dx, Dy, Dz)
# ==========================

def zscore_normalization(data):
    """Z-score normalize inside brain (non-zero) and keep background = 0."""
    brain = data[data > 0]
    if brain.size == 0:
        return data

    mean = brain.mean()
    std = brain.std()
    if std < 1e-6:
        std = 1e-6

    norm = (data - mean) / std
    norm[data == 0] = 0
    return norm

def get_brain_bbox(flair_data, seg_data, margin=0):
    """Bounding box from union of non-zero in flair or seg."""
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

def pad_to_shape(volume, target_shape):
    """
    Center-pad a 3D volume to target_shape with zeros.
    volume: (dx, dy, dz)
    target_shape: (Dx, Dy, Dz)
    """
    dx, dy, dz = volume.shape
    Dx, Dy, Dz = target_shape

    if dx > Dx or dy > Dy or dz > Dz:
        raise ValueError(f"Volume {volume.shape} larger than target {target_shape}")

    padded = np.zeros(target_shape, dtype=volume.dtype)

    off_x = (Dx - dx) // 2
    off_y = (Dy - dy) // 2
    off_z = (Dz - dz) // 2

    padded[off_x:off_x + dx,
           off_y:off_y + dy,
           off_z:off_z + dz] = volume
    return padded

def process_case(case_dir, rel_dir):
    flair_file, seg_file = None, None

    for f in os.listdir(case_dir):
        if f.endswith("_flair.nii") or f.endswith("_flair.nii.gz"):
            flair_file = f
        elif f.endswith("_seg.nii") or f.endswith("_seg.nii.gz"):
            seg_file = f

    if flair_file is None or seg_file is None:
        print(f"[WARN] Missing flair/seg in {case_dir}, skipping.")
        return

    flair_path = os.path.join(case_dir, flair_file)
    seg_path = os.path.join(case_dir, seg_file)

    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)

    flair_data = flair_img.get_fdata()
    seg_data = seg_img.get_fdata()

    bbox = get_brain_bbox(flair_data, seg_data)
    if bbox is None:
        print(f"[WARN] Empty brain mask in {case_dir}, skipping.")
        return

    xs, ys, zs = bbox
    flair_crop = flair_data[xs, ys, zs]
    seg_crop = seg_data[xs, ys, zs]

    # Normalize cropped FLAIR
    flair_norm = zscore_normalization(flair_crop)

    # Pad both to TARGET_SHAPE
    flair_padded = pad_to_shape(flair_norm, TARGET_SHAPE)
    seg_padded = pad_to_shape(seg_crop, TARGET_SHAPE)

    out_case_dir = os.path.join(OUTPUT_ROOT, rel_dir)
    os.makedirs(out_case_dir, exist_ok=True)

    out_flair_path = os.path.join(out_case_dir, flair_file)
    out_seg_path = os.path.join(out_case_dir, seg_file)

    nib.save(nib.Nifti1Image(flair_padded, flair_img.affine, flair_img.header),
             out_flair_path)
    nib.save(nib.Nifti1Image(seg_padded, seg_img.affine, seg_img.header),
             out_seg_path)

    print(f"[OK] {rel_dir}: crop {flair_crop.shape} -> padded {flair_padded.shape}")

def main():
    for root, dirs, files in os.walk(INPUT_ROOT):
        has_nifti = any(
            f.endswith("_flair.nii") or f.endswith("_flair.nii.gz") or
            f.endswith("_seg.nii") or f.endswith("_seg.nii.gz")
            for f in files
        )
        if not has_nifti:
            continue

        rel_dir = os.path.relpath(root, INPUT_ROOT)
        process_case(root, rel_dir)

if __name__ == "__main__":
    main()
