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
# label_ids = ["01", "03", "04", "05", "07"]
label_ids = ["01", "02", "03", "04", "05", "06", "07", "08"]
train_val = (.8, .2)
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
                               partition_ratios=train_val)

# %% Training
model_training = ModelTraining(model, output_dir=model_output_dir, model_name=model_output_name,
                               log_name=training_log_name, train_path=train_path, val_path=val_path,
                               learn_rate=learn_rate, weight_decay=weight_decay, num_epochs=num_epochs,
                               train_batch_size=train_batch_size, val_batch_size=val_batch_size, size=size)

# %% Finalising
if not keep_processed_data and os.path.exists(output_dir):
    shutil.rmtree(output_dir)
