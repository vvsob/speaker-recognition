from datasets import load_dataset
import torchaudio

from dataset import DatasetWrapper

def main():
    cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="train[:100]", data_dir="dataset")
    ds = DatasetWrapper(cv_11, p_noise=0.0, p_smooth=0.0, p_resample=0.0, max_noise_intensity=0.02, smoothness_factor=40, min_resample=3000, max_resample=8000)

    # t = datetime.now()
    # for i in tqdm(range(10000)):
    #     ds[random.randint(0, len(cv_11)-1)]
    # t2 = datetime.now()
    # print(t2 - t)
    torchaudio.save(f"ideal.wav", ds[0]["array"], ds[0]["sampling_rate"])

if __name__ == "__main__":
    main()