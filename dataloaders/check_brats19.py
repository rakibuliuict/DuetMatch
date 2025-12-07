from brats19 import BraTS

base_dir = r'D:\Dataset\BraTS_2019\BraTS_Flair_seg\BraTS_2019'  # The root directory where 'data', 'train.text', 'val.text', and 'test.text' are located



# Load the training set
train_dataset = BraTS(base_dir=base_dir, split='train', transform=None)

# Load the validation set
val_dataset = BraTS(base_dir=base_dir, split='val', transform=None)
