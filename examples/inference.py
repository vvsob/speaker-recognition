import torch
import torchaudio
from model.predictor import Predictor

predictor = Predictor(6, input("Model checkpoint path: "))
waveform, sample_rate = torchaudio.load(input("Sound file location: "))

waveform = waveform[0]
step = 5 * sample_rate

for start in range(0, waveform.shape[0], step):
    frag = waveform[start:start+step]
    pred = predictor.predict(frag, sample_rate)
    print(int(pred.argmax(dim=0)), pred)
