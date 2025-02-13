import torch
import torchaudio

from dataset import DatasetWrapper
from model.model import VoicePredictor
import datasets
from model.trainer import extract_features, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch, 'mps') and torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = VoicePredictor(num_classes=6).to(device)
checkpoint = torch.load(input("Model path file: "), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

test_ds = DatasetWrapper(datasets.load_from_disk("dataset/test"), p_random_noise=0, p_smooth=0, p_resample=0)

batch_size = 4
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

all_labels = []
all_predicts = []

with torch.no_grad():
    for data in tqdm(test_loader):
        inputs = data['array']
        labels = data['speaker_id']

        inputs = extract_features(list(inputs))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.tolist())
        all_predicts.extend(predicted.tolist())

print(all_labels)
print(all_predicts)

disp_labels = ["Slava", "Misha", "Oleg", "Bogdan", "Dima", "other"]

cm = confusion_matrix(all_labels, all_predicts)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
