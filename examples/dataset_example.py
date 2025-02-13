from datasets import load_dataset
import torchaudio

from dataset import DatasetWrapper
from datetime import datetime
from tqdm import tqdm
import random


def main():
    cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="train[:100]", data_dir="dataset")
    ds = DatasetWrapper(cv_11, p_random_noise=0.1, p_smooth=0.1, p_resample=0.1, max_noise_intensity=0.02,
                        min_smoothness_factor=20, max_smoothness_factor=100, smoothness_factors_step=20,
                        min_resample=4000, max_resample=8000)

    # t1 = datetime.now()
    # for i in tqdm(range(1000)):
    #     ds[random.randint(0, len(cv_11) - 1)]
    # t2 = datetime.now()
    # print(f"Execution time for 1000 examples: {t2 - t1}")
    torchaudio.save(f"{ds[0]['speaker_id']}.wav", ds[0]["array"], ds[0]["sampling_rate"])


if __name__ == "__main__":
    main()
