import torch
import torchaudio
from model.predictor import Predictor
from noisereduce.torchgate import TorchGate


predictor = Predictor(6, input("Model checkpoint path: "))
waveform, sample_rate = torchaudio.load(input("Sound file location: "))
tg = TorchGate(sr=sample_rate, nonstationary=False)
waveform = tg(waveform)

torchaudio.save("test_voices/output.wav", waveform, sample_rate)

waveform = waveform[0]
step = 5 * sample_rate

for start in range(0, waveform.shape[0], step):
    frag = waveform[start:start+step]
    pred = predictor.predict(frag, sample_rate)
    print(int(pred.argmax(dim=0)), pred)
