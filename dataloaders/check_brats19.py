


import os
import glob

def check_dataset(root_path):
    """
    This function checks if the .h5 files exist for each patient in the dataset directory.
    
    :param root_path: Root path of the dataset (e.g., '/content/drive/MyDrive/Research/Dataset/data/brats2019')
    """
    
    # Get the list of all patient directories (assuming they are at the root of `root_path`)
    patient_dirs = glob.glob(os.path.join(root_path, 'data', '*'))
    
    # Initialize a counter for missing files
    missing_files = 0

    for patient_dir in patient_dirs:
        # Extract the patient name from the directory
        patient_name = os.path.basename(patient_dir)

        # Get the list of all .h5 files for this patient
        h5_files = glob.glob(os.path.join(patient_dir, '*.h5'))

        # Check if there are any .h5 files
        if not h5_files:
            print(f"Warning: No .h5 files found for patient {patient_name}. Check the directory: {patient_dir}")
            missing_files += 1
        else:
            print(f"Patient {patient_name} has {len(h5_files)} .h5 files.")

    if missing_files == 0:
        print("All patients have .h5 files.")
    else:
        print(f"Total {missing_files} patients are missing .h5 files.")

if __name__ == "__main__":
    # Specify the root path to your dataset (this should be the same root path you're using in your training script)
    root_path = r'D:\Dataset\BraTS_2019\BraTS_Flair_seg\BraTS_2019'
    
    check_dataset(root_path)
