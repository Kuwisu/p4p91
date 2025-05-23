# %% Imports
import argparse
import os
import shutil

import cv2
import librosa
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
import splitfolders

# %% Constants
RAVDESS_EMOTION_MAPPING = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise"
}


# %% Spectrogram Saving
class SpectrogramProcessor:
    def __init__(self,
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = None,
                 n_mels: int = 80,
                 window: str = 'hann',
                 cmap: str = 'gray'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.window = window
        self.cmap = cmap

    def save_spectrogram(self, input_path, save_dir):
        try:
            # Load audio file
            y, _ = librosa.load(input_path, sr=self.sr)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None

        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft,
                                                  hop_length=self.hop_length, win_length=self.win_length,
                                                  n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4), frameon=False)
        librosa.display.specshow(log_mel_spec, sr=self.sr, hop_length=self.hop_length, cmap=self.cmap)

        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(input_path))[0] + ".png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
        plt.close()  # Close the plot to avoid overlapping

        return save_path


# %% File Processing
class FileProcessor:
    def __init__(self,
                 spectrogram_processor: SpectrogramProcessor,
                 input_dir: str = "input-data",
                 output_dir: str = "processed-data",
                 label_ids: list[str] = None,
                 partition_ratios: tuple[float, float] = (.8, .2)
                 ):
        if label_ids is None:
            label_ids = ["01", "03", "04", "05", "07"]

        input_dir = os.path.join(os.getcwd(), input_dir)
        files = os.listdir(input_dir)

        midpoint_dir = "temp"
        os.makedirs(midpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) > 2:
                    emotion_code = parts[2]
                    if (emotion_code in RAVDESS_EMOTION_MAPPING.keys() and
                            emotion_code in label_ids):
                        label = RAVDESS_EMOTION_MAPPING[emotion_code]
                        filepath = os.path.join(input_dir, file)

                        # Save the spectrogram and create a new directory if it does not exist
                        save_dir = os.path.join(midpoint_dir, label)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = spectrogram_processor.save_spectrogram(filepath, save_dir)
                        print(f"Saved spectrogram to {save_path}")

        # Split into testing, training, and validation sets and delete the intermediate folder
        splitfolders.ratio(midpoint_dir, output_dir, seed=42, ratio=partition_ratios)
        if os.path.exists(midpoint_dir):
            shutil.rmtree(midpoint_dir)


# %% Implementation for running script on its own
if __name__ == "__main__":
    spectrogram_processor = SpectrogramProcessor()
    processFiles = FileProcessor(spectrogram_processor)
