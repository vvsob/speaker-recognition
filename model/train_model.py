import torchaudio

from dataset import DatasetWrapper
import datasets

from model.voice_predictor import *
from model.trainer import *

from torcheval.metrics import MulticlassAccuracy


cnt_users = 6

names_to_id = {
    "slava": 0,
    "misha": 1,
    "oleg": 2,
    "bogdan": 3,
    "dima": 4,
    "artem": 5,
    "lesha": 5,
    "nastya": 5,
    "other-1": 5,
    "other-2": 5
}


if not os.path.exists("dataset"):
    os.mkdir("dataset")

    ds_train_list = []
    ds_test_list = []

    for user_name, client_id in names_to_id.items():
        if not os.path.exists(f"voices/{user_name}"):
            continue

        print(f"Processing {user_name}")

        dirname = f"voices/{user_name}"
        for file in os.listdir(f"voices/{user_name}"):
            print(f"-> {file}")

            waveform, sample_rate = torchaudio.load(f"{dirname}/{file}")
            waveform = waveform[0]

            step = 5 * sample_rate

            wf_max = waveform.max()

            start = 0
            cnt_iters = 0

            while start < waveform.shape[0]:
                if cnt_iters % 5 == 0:
                    frag = waveform[start:start + step]

                    ds_test_list.append({
                        'audio': {
                            'array': frag,
                            'sampling_rate': sample_rate
                        },
                        'client_id': client_id
                    })

                    start += step
                else:
                    frag = waveform[start:start + 2 * step]

                    ds_train_list.append({
                        'audio': {
                            'array': frag,
                            'sampling_rate': sample_rate
                        },
                        'client_id': client_id
                    })

                    start += 2 * step

                cnt_iters += 1

    train_ds = datasets.Dataset.from_list(ds_train_list)
    test_ds = datasets.Dataset.from_list(ds_test_list)

    train_ds.save_to_disk("dataset/train")
    test_ds.save_to_disk("dataset/test")
else:
    train_ds = datasets.load_from_disk("dataset/train")
    test_ds = datasets.load_from_disk("dataset/test")


print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")


train_ds = DatasetWrapper(train_ds, p_random_noise=0.15, p_smooth=0.15, p_resample=0.15, p_real_noise=0.4, min_fragment_length=4, max_fragment_length=6, noise_dir="noise")
test_ds = DatasetWrapper(test_ds, p_random_noise=0, p_smooth=0, p_resample=0)

model = VoicePredictor(cnt_users)

trainer = Trainer(model, train_ds, test_ds, 16)
trainer.fit('Adam', nn.CrossEntropyLoss(), num_epochs=200, metrics=[MulticlassAccuracy()], checkpoint_metric=MulticlassAccuracy())
