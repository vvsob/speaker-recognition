from datetime import datetime

from datasets import Dataset, load_dataset
import random
import torch, torchaudio
import torch.nn.functional as F
import torchaudio.functional as FF

from tqdm import tqdm


class DatasetWrapper:
    def __init__(self, data, sampling_rate=16000, name=None, p_noise=0.0, p_smooth=0.0, p_resample=0.0, max_noise_intensity=0.02, smoothness_factor=20, min_resample=2000, max_resample=8000 ):
        """
        Initializes the HuggingFaceDatasetWrapper.  Assumes input is always a Dataset.

        Args:
            data: The Hugging Face Dataset object.
            name (str, optional): A name for the dataset. Defaults to None.
        """

        if not isinstance(data, Dataset):
            raise TypeError("Data must be a Hugging Face Dataset object.")

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
        if random.random() < self.p_noise:
            resulting_noise_intensity = (self.noise_intensity * item["array"].abs().max() * random.random())
            noise = torch.randn_like(item["array"]) * resulting_noise_intensity
            item["array"] = item["array"] + noise
        if random.random() < self.p_smooth:
            item["array"] = F.conv1d(item["array"], self.smoothness_kernel, padding=self.smoothness_padding).squeeze().unsqueeze(0)
        if random.random() < self.p_resample:
            new_sampling_rate = random.choice(self.random_sampling_rates)
            item["array"] = FF.resample(item["array"], item["sampling_rate"], new_sampling_rate)
            item["sampling_rate"] = new_sampling_rate
        item["array"] = FF.resample(item["array"], item["sampling_rate"], self.sampling_rate)
        item["sampling_rate"] = self.sampling_rate
        return item

    def __getitem__(self, idx):
        raw_item = self.dataset[idx]
        item = {"array": torch.FloatTensor(raw_item["audio"]["array"]).unsqueeze(0), "sampling_rate": raw_item["audio"]["sampling_rate"],
                "speaker_id": raw_item["client_id"]}
        return self.augmentate(item)  # Apply augmentation

    def save_to_disk(self, path, *args, **kwargs):
        return self.dataset.save_to_disk(path, *args, **kwargs)

    @classmethod
    def from_disk(cls, path, *args, **kwargs):
        dataset = Dataset.from_disk(path, *args, **kwargs)
        return cls(dataset)


def main():
    cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="train[:100]", data_dir="dataset")
    ds = DatasetWrapper(cv_11, p_noise=0.0, p_smooth=0.0, p_resample=0.0, max_noise_intensity=0.02, smoothness_factor=20, min_resample=2000, max_resample=4000)

    # t = datetime.now()
    # for i in tqdm(range(10000)):
    #     ds[random.randint(0, len(cv_11)-1)]
    # t2 = datetime.now()
    # print(t2 - t)
    torchaudio.save(f"ideal.wav", ds[0]["array"], ds[0]["sampling_rate"])

if __name__ == "__main__":
    main()
