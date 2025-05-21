# %% Imports
from preprocessing import *
from resnetmodel import *

import numpy as np

from training import ModelTraining

# %% Spectrogram parameters
sr = 16000
n_fft = 2048
hop_length = 512
win_length = 2048
n_mels = 80
window = 'hann'
cmap = 'gray'

# %% Preprocessing parameters
input_dir = "input-data"
output_dir = "processed-data"
# View preprocessing.RAVDESS_EMOTION_MAPPING to see options
label_ids = ["01", "03", "04", "05", "07"]
train_val = (.8, .2)
# If False, uses [0.5], [0.5] for normalise mean and standard deviation
use_dataset_mean_std = False
keep_processed_data = True

# %% Training parameters
train_path = output_dir + "/train"
val_path = output_dir + "/val"
model_output_dir = "model-out"
training_log_name = "training-log.txt"
model_output_name = "p4p91-emotion-resnet"
weight_decay = 1e-5
learn_rate = 0.001
num_epochs = 50
size = (224, 224)
train_batch_size = 32
val_batch_size = 32
model = ResNet50(num_classes=len(label_ids))

# %% Pipeline
spectrogram_processor = SpectrogramProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length,
                                             win_length=win_length, n_mels=n_mels,
                                             window=window, cmap=cmap)
file_processor = FileProcessor(spectrogram_processor, input_dir=input_dir, output_dir=output_dir, label_ids=label_ids,
                               calculate_mean_std=use_dataset_mean_std, partition_ratios=train_val)

if use_dataset_mean_std:
    std = file_processor.std
    mean = file_processor.mean
else:
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])

# %% Training
model_training = ModelTraining(model, output_dir=model_output_dir, model_name=model_output_name,
                               log_name=training_log_name, train_path=train_path, val_path=val_path,
                               learn_rate=learn_rate, weight_decay=weight_decay, num_epochs=num_epochs,
                               train_batch_size=train_batch_size, val_batch_size=val_batch_size,
                               mean=mean, std=std, size=size)

# %% Finalising
if not keep_processed_data and os.path.exists(output_dir):
    shutil.rmtree(output_dir)
