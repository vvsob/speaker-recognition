import torch
import torchaudio
from model.model import VoicePredictor
from transformers import Wav2Vec2FeatureExtractor


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch, 'mps') and torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

waveform, sample_rate = torchaudio.load('/Users/chertan/PycharmProjects/speaker-recognition/voices/slava/Слава_pixel.wav')
waveform = waveform[0]

mid = waveform.shape[0]
offset = 5 * sample_rate // 2

frag = waveform[mid-offset:mid+offset]
frag_16000 = torchaudio.functional.resample(frag, sample_rate, 16000)

model = VoicePredictor(num_classes=6).to(device)
checkpoint = torch.load("/Users/chertan/Downloads/second_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

inputs = feature_extractor(
    [frag_16000],
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=16000 * 5
)

inp = inputs['input_values'][0].to(device)

print(torch.softmax(model(inp.to(torch.device('mps'))), dim=1).tolist())

