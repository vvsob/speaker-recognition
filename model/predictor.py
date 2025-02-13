import torch
import torchaudio
from model.voice_predictor import VoicePredictor
from model.trainer import extract_features


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
        self.model = VoicePredictor(num_classes=num_classes).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()

    def predict(self, waveform: torch.Tensor, sample_rate) -> torch.Tensor:
        with torch.no_grad():
            if len(waveform.shape) > 1:
                waveform = waveform[0]

            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            inp = extract_features([waveform]).to(self.device)

            predicts = torch.softmax(self.model(inp.to(self.device)), dim=1)[0]

        return predicts
