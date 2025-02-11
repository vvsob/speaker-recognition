from datasets import load_dataset
import torchaudio
import torch

from dataset import DatasetWrapper
from datetime import datetime
from tqdm import tqdm
import random
from tqdm import tqdm
from dataset import DatasetWrapper

from model.model import *
from model.trainer import *

from torcheval.metrics import MulticlassAccuracy

cv_11 = load_dataset("mozilla-foundation/common_voice_11_0", "ru", split="train[:1000]", data_dir="dataset")

users_records = dict()

for i in tqdm(range(len(cv_11))):
    if cv_11[i]['client_id'] not in users_records.keys():
        users_records[cv_11[i]['client_id']] = 0

    users_records[cv_11[i]['client_id']] += 1

pairs = []

for key, value in users_records.items():
    pairs.append((key, value))

pairs.sort(key=lambda x: -x[1])

cnt_users = min(5, len(pairs))

hash_to_id = dict()
for i in range(cnt_users):
    hash_to_id[pairs[i][0]] = i

cv_11 = cv_11.filter(lambda x: x['client_id'] in hash_to_id.keys())


def update_ids(row):
    row['client_id'] = hash_to_id[row['client_id']]
    return row


cv_11 = cv_11.map(update_ids)

split_dataset = cv_11.train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=52
)

train_ds = DatasetWrapper(split_dataset['train'], p_noise=0.1, p_smooth=0.1, p_resample=0.1, max_noise_intensity=0.02,
                    smoothness_factor=40, min_resample=4000, max_resample=8000)

test_ds = DatasetWrapper(split_dataset['test'], p_noise=0, p_smooth=0, p_resample=0, max_noise_intensity=0,
                    smoothness_factor=0, min_resample=0, max_resample=0)

model = VoicePredictor(cnt_users)

trainer = Trainer(model, train_ds, test_ds, 4)
trainer.fit('Adam', nn.CrossEntropyLoss(), metrics=[MulticlassAccuracy()])
