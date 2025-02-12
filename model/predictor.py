import torch
import torchaudio
from model.model import VoicePredictor
from transformers import Wav2Vec2FeatureExtractor


class Predictor:
    def __init__(self, num_classes, checkpoint_path, device=None):
        if not device:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = VoicePredictor(num_classes=num_classes).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, waveform: torch.Tensor, sample_rate):
        if len(waveform.shape) > 1:
            waveform = waveform[0]

        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        inputs = self.feature_extractor(
            [waveform],
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=16000 * 5
        )

        inp = inputs['input_values'][0].to(self.device)

        predicts = torch.softmax(self.model(inp.to(torch.device('mps'))), dim=1)[0]

        return int(torch.argmax(predicts, dim=0))
