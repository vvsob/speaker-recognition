from datasets import Dataset
import random
import torch
import torch.nn.functional as F
import torchaudio.functional as FF

class DatasetWrapper:
    def __init__(self, data, sampling_rate=16000, name=None, p_noise=0.0, p_smooth=0.0, p_resample=0.0, max_noise_intensity=0.02, smoothness_factor=40, min_resample=4000, max_resample=8000 ):
        """
        Initializes the DatasetWrapper.  Assumes input is always a Dataset.

        Args:
            data: The Hugging Face Dataset object.
            sampling_rate: The target sampling rate of the audio.
            name: The name of the dataset.
            p_noise: The probability of noise.
            p_smooth: The probability of smoothing.
            p_resample: The probability of resampling.
            max_noise_intensity: The maximum intensity of the noise.
            smoothness_factor: The smoothness factor.
            min_resample: The minimum resample rate.
            max_resample: The maximum resample rate.
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

        self.dataset = data
        self.sampling_rate = sampling_rate
        self.name = name
        self.p_noise = p_noise
        self.p_smooth = p_smooth
        self.p_resample = p_resample
        self.noise_intensity = max_noise_intensity
        self.smoothness_factor = smoothness_factor
        self.random_sampling_rates = range(min_resample, max_resample + 1, 1000)

        factors = [1.0]
        for i in range(self.smoothness_factor-1):
            new_factors = [factors[0]/2]
            for j in range(len(factors)-1):
                new_factors.append((factors[j] + factors[j+1])/2)
            new_factors.append(factors[0]/2)
            factors=new_factors

        self.smoothness_kernel = torch.tensor(factors, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.smoothness_padding = (self.smoothness_kernel.size(-1) - 1) // 2

    def __len__(self):
        return len(self.dataset)

    def augmentate(self, item):
        array = item["array"]
        sampling_rate = item["sampling_rate"]
        if random.random() < self.p_noise:
            resulting_noise_intensity = (self.noise_intensity * array.abs().max() * random.random())
            noise = torch.randn_like(array) * resulting_noise_intensity
            array = array + noise
        if random.random() < self.p_smooth:
            array = F.conv1d(array, self.smoothness_kernel, padding=self.smoothness_padding).squeeze().unsqueeze(0)
        if random.random() < self.p_resample:
            new_sampling_rate = random.choice(self.random_sampling_rates)
            array = FF.resample(array, sampling_rate, new_sampling_rate)
            sampling_rate = new_sampling_rate
        item["array"] = FF.resample(array, sampling_rate, self.sampling_rate)[:1]
        item["sampling_rate"] = self.sampling_rate
        return item

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        item = {"array": torch.FloatTensor(raw_item["audio"]["array"]).unsqueeze(0), "sampling_rate": raw_item["audio"]["sampling_rate"],
                "speaker_id": raw_item["client_id"]}
        return self.augmentate(item)

    def save_to_disk(self, path, *args, **kwargs):
        return self.dataset.save_to_disk(path, *args, **kwargs)

    @classmethod
    def load_from_disk(cls, path, *args, **kwargs):
        dataset = Dataset.load_from_disk(path, *args, **kwargs)
        return cls(dataset)
