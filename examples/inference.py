import torch
import torchaudio
from model.predictor import Predictor

predictor = Predictor(6, input("Model checkpoint path: "))
waveform, sample_rate = torchaudio.load(input("Sound file location: "))

mid = waveform.shape[1]
offset = 5 * sample_rate // 2

frag = waveform[:, mid-offset:mid+offset]

print(predictor.predict(frag, sample_rate))
