from datasets import Dataset
import random
import torch
import torch.nn.functional as F
import torchaudio.functional as FF
import os
import torchaudio


class DatasetWrapper:
    def __init__(self, data, sampling_rate=16000, name=None, p_random_noise=0.0, p_real_noise=0.0, p_smooth=0.0, p_resample=0.0,
                 max_noise_intensity=0.02, min_smoothness_factor=20,
                 max_smoothness_factor=100, smoothness_factors_step=10, min_resample=4000, max_resample=8000,
                 min_fragment_length=None, max_fragment_length=None, noise_dir=None):
        """
        Initializes the DatasetWrapper.  Assumes input is always a Dataset or a list of dictionaries.

        Args:
            data: The Hugging Face Dataset object or a list of dictionaries.
            sampling_rate: The target sampling rate of the audio.
            name: The name of the dataset.
            p_random_noise: The probability of random normal distributed noise.
            p_real_noise: The probability of noise loaded from files
            p_smooth: The probability of smoothing.
            p_resample: The probability of resampling.
            max_noise_intensity: The maximum intensity of the noise.
            min_smoothness_factor: The minimum smoothness factor.
            max_smoothness_factor: The maximum smoothness factor.
            smoothness_factors_step: Regulates the step of kernels dimensions for smoothing sound.
            min_resample: The minimum resample rate.
            max_resample: The maximum resample rate.
            min_fragment_length: The minimum fragment length.
            max_fragment_length: The maximum fragment length.
            noise_dir: directory where stored noise .wav files
        """

        if isinstance(data, list):
            # Convert list of dictionaries to Hugging Face Dataset
            dataset_dict = {
                "audio": [{"array": item["audio"]["array"], "sampling_rate": item["audio"]["sampling_rate"]} for item in
                          data],
                "client_id": [item.get("client_id", None) for item in data]  # Handles missing client_id
            }
            self.dataset = Dataset.from_dict(dataset_dict)
        elif isinstance(data, Dataset):
            self.dataset = data
        else:
            raise TypeError("Data must be a Hugging Face Dataset object or a list of dictionaries.")

        self.sampling_rate = sampling_rate
        self.name = name
        self.p_random_noise = p_random_noise
        self.p_real_noise = p_real_noise
        self.p_smooth = p_smooth
        self.p_resample = p_resample
        self.noise_intensity = max_noise_intensity
        self.random_sampling_rates = range(min_resample, max_resample + 1, 1000)
        self.kernels = [self.get_kernel(smoothness_factor) for smoothness_factor in
                        range(min_smoothness_factor, max_smoothness_factor + 1, smoothness_factors_step)]

        self.min_fragment_length = min_fragment_length
        self.max_fragment_length = max_fragment_length

        self.noises = []

        if noise_dir:
            for file in os.listdir(noise_dir):
                try:
                    noise_wf, noise_sr = torchaudio.load(f"{noise_dir}/{file}")
                except:  # skip if can't load file
                    continue

                noise_wf = FF.resample(noise_wf, noise_sr, sampling_rate)
                self.noises.append(noise_wf[0])

    def get_kernel(self, smoothness_factor):
        """Count the binomial coefficients"""
        factors = [1.0]
        for i in range(smoothness_factor - 1):
            new_factors = [factors[0] / 2]
            for j in range(len(factors) - 1):
                new_factors.append((factors[j] + factors[j + 1]) / 2)
            new_factors.append(factors[0] / 2)
            factors = new_factors
        kernel = torch.tensor(factors, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        padding = (kernel.size(-1) - 1) // 2
        return (kernel, padding)

    def __len__(self):
        return len(self.dataset)

    def augmentate(self, item):
        array = item["array"]
        sampling_rate = item["sampling_rate"]
        if random.random() < self.p_random_noise:
            resulting_noise_intensity = (self.noise_intensity * array.abs().max() * random.random())
            noise = torch.randn_like(array) * resulting_noise_intensity
            array = array + noise
        if random.random() < self.p_smooth:
            kernel, padding = random.choice(self.kernels)
            array = F.conv1d(array, kernel, padding=padding).squeeze().unsqueeze(0)
        if random.random() < self.p_resample:
            new_sampling_rate = random.choice(self.random_sampling_rates)
            array = FF.resample(array, sampling_rate, new_sampling_rate)
            sampling_rate = new_sampling_rate
        if random.random() < self.p_real_noise:
            total_length = sum(map(lambda x: x.shape[0], self.noises))

            val = random.randint(0, total_length - 1)

            prev_len = 0
            for i in range(len(self.noises)):  # choose one of noise files according to their length
                if prev_len <= val < prev_len + self.noises[i].shape[0]:
                    start = random.randint(0, self.noises[i].shape[0] - item["array"].shape[-1])
                    noise = self.noises[i][start:start+item["array"].shape[-1]]

                    noise = torch.stack([noise] * item["array"].shape[0])  # from 1D to 2D
                    item["array"] += noise

                    break

                prev_len += self.noises[i].shape[0]

        item["array"] = FF.resample(array, sampling_rate, self.sampling_rate)[:1]
        item["sampling_rate"] = self.sampling_rate

        if self.min_fragment_length and self.max_fragment_length:  # get random segment of wav
            fragment_length = int(self.sampling_rate * (self.min_fragment_length + random.random() * (
                        self.max_fragment_length - self.min_fragment_length)))
            fragment_length = min(fragment_length, item["array"].shape[1])
            start = random.randint(0, item["array"].shape[1] - fragment_length)

            item["array"] = item["array"][:, start:start + fragment_length]

        return item

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        item = {"array": torch.FloatTensor(raw_item["audio"]["array"]).unsqueeze(0),
                "sampling_rate": raw_item["audio"]["sampling_rate"],
                "speaker_id": raw_item["client_id"]}
        return self.augmentate(item)

    def save_to_disk(self, path, *args, **kwargs):
        return self.dataset.save_to_disk(path, *args, **kwargs)

    @classmethod
    def load_from_disk(cls, path, *args, **kwargs):
        dataset = Dataset.load_from_disk(path, *args, **kwargs)
        return cls(dataset)
