import os.path

from datasets import load_dataset
import torchaudio
import torch

from dataset import DatasetWrapper
from datetime import datetime
from tqdm import tqdm
import random
from dataset import DatasetWrapper
import datasets
import sys

from model.model import *
from model.trainer import *

from torcheval.metrics import MulticlassAccuracy
from model.metrics import MulticlassF1Score

import numpy as np

cnt_users = 6

names_to_id = {
    "slava": 0,
    "misha": 1,
    "oleg": 2,
    "bogdan": 3,
    "dima": 4,
    "artem": 5,
    "lesha": 5,
    "nastya": 5
}

if not os.path.exists("dataset"):
    os.mkdir("dataset")

    ds_train_list = []
    ds_test_list = []

    for user_name, client_id in names_to_id.items():
        print(f"Processing {user_name}")

        dirname = f"voices/{user_name}"
        for file in os.listdir(f"voices/{user_name}"):
            print(f"-> {file}")

            waveform, sample_rate = torchaudio.load(f"{dirname}/{file}")
            waveform = waveform[0]

            step = 10 * sample_rate

            wf_max = waveform.max()

            for start in range(0, waveform.shape[0], step):
                frag = waveform[start:start+step]

                if frag.max() / wf_max > 0.4:
                    if start / waveform.shape[0] < 0.2:
                        ds_test_list.append({
                            'audio': {
                                'array': frag,
                                'sampling_rate': sample_rate
                            },
                            'client_id': client_id
                        })
                    else:
                        ds_train_list.append({
                            'audio': {
                                'array': frag,
                                'sampling_rate': sample_rate
                            },
                            'client_id': client_id
                        })
    train_ds = datasets.Dataset.from_list(ds_train_list)
    test_ds = datasets.Dataset.from_list(ds_test_list)

    train_ds.save_to_disk("dataset/train")
    test_ds.save_to_disk("dataset/test")
else:
    train_ds = datasets.load_from_disk("dataset/train")
    test_ds = datasets.load_from_disk("dataset/test")


print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")


train_ds = DatasetWrapper(train_ds, p_noise=0.5, p_smooth=0.5, p_resample=0.5, min_fragment_length=4, max_fragment_length=6)
test_ds = DatasetWrapper(test_ds, p_noise=0, p_smooth=0, p_resample=0)

model = VoicePredictor(cnt_users)

trainer = Trainer(model, train_ds, test_ds, 16)
trainer.fit('Adam', nn.CrossEntropyLoss(), num_epochs=200, metrics=[MulticlassAccuracy()], checkpoint_metric=MulticlassAccuracy())
