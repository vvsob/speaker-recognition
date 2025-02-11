from transformers import HubertForSequenceClassification
from torch import nn


class VoicePredictor(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ser = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
        self.ser.projector = nn.Identity()
        self.ser.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb = self.ser(x).logits
        out = self.classifier(emb)

        return out
