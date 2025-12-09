import os
import glob
import numpy as np
import SimpleITK as sitk
import h5py


def find_file(case_dir, suffixes):
    """
    Find exactly one file in case_dir with any of the suffixes.
    Example suffixes: ["_flair.nii.gz", "_flair.nii"]
    """
    for suf in suffixes:
        files = glob.glob(os.path.join(case_dir, "*" + suf))
        if len(files) == 1:
            return files[0]
    return None


def create_h5_for_case(case_dir):
    """
    For a given case folder:
        - load flair and seg
        - create <case_name>.h5 in the same folder
        - store datasets: 'flair', 'seg'
    """
    case_name = os.path.basename(case_dir.rstrip("/"))

    # -----------------------
    # Locate flair and seg
    # -----------------------
    flair_path = find_file(case_dir, ["_flair.nii.gz", "_flair.nii"])
    seg_path   = find_file(case_dir, ["_seg.nii.gz", "_seg.nii"])

    if flair_path is None or seg_path is None:
        print(f"[WARN] Missing flair or seg in: {case_dir}")
        return

    # -----------------------
    # Read NIfTI using SimpleITK
    # -----------------------
    flair_img = sitk.ReadImage(flair_path)
    seg_img   = sitk.ReadImage(seg_path)

    flair_np = sitk.GetArrayFromImage(flair_img)  # [Z, Y, X]
    seg_np   = sitk.GetArrayFromImage(seg_img)    # [Z, Y, X]

    if flair_np.shape != seg_np.shape:
        print(f"[WARN] Shape mismatch in {case_dir}: "
              f"flair {flair_np.shape}, seg {seg_np.shape}")
        return

    # Optional: cast to consistent dtypes for HDF5
    flair_np = flair_np.astype(np.float32)
    seg_np   = seg_np.astype(np.uint8)

    # -----------------------
    # Create .h5 file
    # -----------------------
    h5_path = os.path.join(case_dir, f"{case_name}.h5")

    with h5py.File(h5_path, "w") as f:
        # Store volumes
        f.create_dataset("image", data=flair_np, compression="gzip")
        f.create_dataset("label",   data=seg_np,   compression="gzip")

        # Store some metadata as attributes
        f.attrs["case_name"] = case_name
        f.attrs["spacing"]   = flair_img.GetSpacing()
        f.attrs["origin"]    = flair_img.GetOrigin()
        f.attrs["direction"] = flair_img.GetDirection()

    print(f"[OK] Created HDF5: {h5_path}  shape={flair_np.shape}")


def create_h5_for_root(root_dir):
    """
    root_dir contains folders like:
        BraTS19_TCIA13_624_1/
        BraTS19_TCIA13_630_1/
        ...

    For each folder, create a <foldername>.h5 beside flair/seg.
    """
    case_dirs = sorted(
        d for d in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(d)
    )

    print("Found", len(case_dirs), "case folders")

    for case_dir in case_dirs:
        create_h5_for_case(case_dir)


if __name__ == "__main__":
    ROOT_DIR = r"D:\Dataset\BraTS_2019\Crop_Flair"   # <<< put your root folder here
    create_h5_for_root(ROOT_DIR)
