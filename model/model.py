from transformers import Wav2Vec2ForSequenceClassification
from torch import nn


class VoicePredictor(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ser = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        self.ser.projector = nn.Identity()
        self.ser.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb = self.ser(x).logits
        out = self.classifier(emb)

        return out
