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
                 fmax: float = None,
                 window: str = 'hann',
                 cmap: str = 'gray'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmax = fmax
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
                                                  n_mels=self.n_mels, fmax=self.fmax)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4))
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
                 calculate_mean_std: bool = True,
                 partition_ratios: tuple[float, float, ...] = (.8, .1, .1)
                 ):
        if label_ids is None:
            label_ids = ["01", "03", "04", "05", "07"]

        input_dir = os.path.join(os.getcwd(), input_dir)
        files = os.listdir(input_dir)

        midpoint_dir = "temp"
        os.makedirs(midpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Initialise arrays for dataset mean and standard deviation calculation
        num_samples = 0
        self.mean = np.array([0., 0., 0.])
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

                        # Update the mean for each image
                        if save_path is not None and calculate_mean_std:
                            num_samples += 1
                            im = cv2.imread(save_path)
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(float) / 255
                            images.append(im)

                            for j in range(3):
                                self.mean[j] += np.mean(im[:, :, j])

        # Calculate the standard deviation with another loop
        stdTemp = np.array([0., 0., 0.])
        self.std = np.array([0., 0., 0.])
        if num_samples > 0 and calculate_mean_std:
            self.mean = self.mean / num_samples
            for image in images:
                for j in range(3):
                    stdTemp[j] += (((image[:, :, j] - self.mean[j]) ** 2).sum() /
                                   (image.shape[0] * image.shape[1]))

            self.std = np.sqrt(stdTemp / num_samples)

        # Split into testing, training, and validation sets and delete the intermediate folder
        splitfolders.ratio(midpoint_dir, output_dir, seed=42, ratio=partition_ratios)
        if os.path.exists(midpoint_dir):
            shutil.rmtree(midpoint_dir)


# %% Implementation for running script on its own
if __name__ == "__main__":
    spectrogram_processor = SpectrogramProcessor()
    processFiles = FileProcessor(spectrogram_processor)

    print(f"Mean of dataset: {processFiles.mean}")
    print(f"Standard deviation of dataset: {processFiles.std}")
