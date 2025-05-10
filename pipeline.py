# %% Imports
from preprocessing import *
from resnetmodel import *

import numpy as np

from training import ModelTraining

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
train_path = input_dir + "/train"
val_path = input_dir + "/val"
test_path = input_dir + "/test"
model_output_dir = "model-out"
model_output_name = "p4p91-emotion-resnet"
# Exclude the third value in the tuple to bypass testing
train_val_test = (.8, .1, .1)
weight_decay = 0
learn_rate = 0.001
num_epochs = 100
size = (224, 224)
train_batch_size = 32
val_batch_size = 32
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
model_training = ModelTraining(model, output_dir=output_dir,
                               train_path=train_path, val_path=val_path, learn_rate=learn_rate,
                               weight_decay=weight_decay, num_epochs=num_epochs, mean=mean, std=std,
                               size=size, train_batch_size=train_batch_size, val_batch_size=val_batch_size)

# %% Testing
# Only do testing if a testing set was partitioned
if len(train_val_test) == 3:
    pass

# %% Finalising
if not keep_processed_data and os.path.exists(output_dir):
    shutil.rmtree(output_dir)
