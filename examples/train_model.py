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

from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

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
    ds_list = []

    for user_name, client_id in names_to_id.items():
        print(f"Processing {user_name}")

        dirname = f"voices/{user_name}"
        for file in os.listdir(f"voices/{user_name}"):
            print(f"-> {file}")

            waveform, sample_rate = torchaudio.load(f"{dirname}/{file}")
            waveform = waveform[0]

            step = 5 * sample_rate

            wf_max = waveform.max()

            for start in range(0, waveform.shape[0], step):
                frag = waveform[start:start+step]

                if frag.max() / wf_max > 0.4:
                    ds_list.append({
                        'audio': {
                            'array': frag,
                            'sampling_rate': sample_rate
                        },
                        'client_id': client_id
                    })

    ds = datasets.Dataset.from_list(ds_list)
    ds.save_to_disk("dataset")
else:
    ds = datasets.load_from_disk("dataset")

split_dataset = ds.train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=52
)


train_ds = DatasetWrapper(split_dataset['train'], p_noise=0.3, p_smooth=0.3, p_resample=0.3, max_noise_intensity=0.02,
                    smoothness_factor=40, min_resample=4000, max_resample=8000)

test_ds = DatasetWrapper(split_dataset['test'], p_noise=0, p_smooth=0, p_resample=0, max_noise_intensity=0,
                    smoothness_factor=0, min_resample=0, max_resample=0)

model = VoicePredictor(cnt_users)

trainer = Trainer(model, train_ds, test_ds, 4)
trainer.fit('Adam', nn.CrossEntropyLoss(), metrics=[MulticlassAccuracy(), MulticlassF1Score()], checkpoint_metric=MulticlassF1Score())
