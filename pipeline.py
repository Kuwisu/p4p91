# %% Imports
from preprocessing import *
from resnetmodel import *

import numpy as np

# %% Spectrogram parameters
sr = 22050
n_fft = 2048
hop_length = 512
win_length = None
n_mels = 80
fmax = None
window = 'hann'
cmap = 'gray'

# %% Preprocessing parameters
input_dir = "input-data"
output_dir = "processed-data"
# View preprocessing.RAVDESS_EMOTION_MAPPING to see options
label_ids = ["01", "03", "04", "05", "07"]
# If False, uses [0.5], [0.5] for normalise mean and standard deviation
use_dataset_mean_std = True
keep_processed_data = True

# %% Training parameters
# Exclude the third value in the tuple to bypass testing
train_val_test = (.8, .1, .1)
weight_decay = 0
model = ResNet50(num_classes=len(label_ids))

# %% Pipeline
spectrogram_processor = SpectrogramProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length,
                                             win_length=win_length, n_mels=n_mels, fmax=fmax,
                                             window=window, cmap=cmap)
file_processor = FileProcessor(spectrogram_processor, input_dir=input_dir, output_dir=output_dir, label_ids=label_ids,
                               calculate_mean_std=use_dataset_mean_std, partition_ratios=train_val_test)

if use_dataset_mean_std:
    std = file_processor.std
    mean = file_processor.mean
else:
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])

# %% Training

# %% Testing
# Only do testing if a testing set was partitioned
if len(train_val_test) == 3:
    pass

# %% Finalising
if not keep_processed_data and os.path.exists(output_dir):
    shutil.rmtree(output_dir)
