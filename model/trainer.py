import json
import os

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
import matplotlib.pyplot as plt


def extract_features(waveforms):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

    TARGET_LENGTH = 16000 * 5  # 80000 samples

    def pad_or_truncate_waveform(waveform, target_length=TARGET_LENGTH):
        """
        Pads or truncates the waveform tensor to a fixed length.
        """
        current_length = waveform.shape[0]

        if current_length < target_length:
            pad_length = target_length - current_length
            waveform = F.pad(waveform, (0, pad_length))  # Right-padding with zeros
        else:
            waveform = waveform[:target_length]  # Truncate if too long

        return waveform

    # Convert list to a single tensor
    waveforms = torch.stack(waveforms)  # Shape: (batch_size, 1, TARGET_LENGTH)

    # Now, pass to the feature extractor
    inputs = feature_extractor(
        waveforms,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=16000 * 5
    )

    return inputs['input_values'][0]


def collate_fn(batch):
    """
    batch: List of tuples (waveform, label).
    Returns padded waveforms and labels.
    """
    waveforms = [item['array'][0] for item in batch]
    sample_rates = [item['sampling_rate'] for item in batch]
    labels = [item['speaker_id'] for item in batch]

    # Find the longest audio in the batch
    max_length = max(waveform.shape[-1] for waveform in waveforms)

    # Pad all waveforms to the max length
    padded_waveforms = [F.pad(w, (0, max_length - w.shape[-1])) for w in waveforms]

    # Stack tensors into a single batch
    padded_waveforms = torch.stack(padded_waveforms)  # Shape: (batch_size, num_samples)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'array': padded_waveforms,
        'sampling_rate': sample_rates,
        'speaker_id': labels
    }


class Trainer:
    def __init__(self, model: nn.Module, train_ds, test_ds, batch_size, device=None, output_dir="training"):
        os.makedirs(f"{output_dir}", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        self.output_dir = output_dir

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')

        self.device = device
        self.model = model.to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print("Total params:", sum(p.numel() for p in self.model.parameters()))
        print("Trainable params:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def fit(self, optimizer, loss, num_epochs=10, lr=0.001, metrics: list = None, checkpoint_metric=None, schedulers: list = None):
        checkpoint_metric = checkpoint_metric.__class__.__name__
        train_metrics_history: list[dict] = []
        test_metrics_history: list[dict] = []

        if isinstance(optimizer, str):  # if optimizer passed as class name
            if hasattr(optim, optimizer):
                optimizer = getattr(optim, optimizer)
            else:
                assert False, "Optimizer not found"

        optimizer = optimizer(self.model.parameters(), lr=lr)

        for metric in metrics:  # reset metrics before loop start and then after every loop inside it
            metric.reset()

        for epoch in range(num_epochs):
            self.model.train()

            loss_train = 0
            total_train = 0

            for data in tqdm(self.train_loader):
                inputs = data['array']
                labels = data['speaker_id']

                inputs = extract_features(list(inputs))
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss_value = loss(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                # Track loss and accuracy
                loss_train += loss_value.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)

                for metric in metrics:
                    metric.update(outputs, labels)

            # calculate train metrics and loss
            train_metrics = {}
            for metric in metrics:
                train_metrics[metric.__class__.__name__] = float(metric.compute())
                metric.reset()

            train_metrics["Loss"] = loss_train / total_train
            train_metrics_history.append(train_metrics)

            # evaluation
            self.model.eval()

            loss_test = 0
            total_test = 0

            with torch.no_grad():
                for data in tqdm(self.test_loader):
                    inputs = data['array']
                    labels = data['speaker_id']

                    inputs = extract_features(list(inputs))
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)

                    loss_value = loss(outputs, labels)

                    loss_test += loss_value.item()
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)

                    for metric in metrics:
                        metric.update(outputs, labels)

            # calculate test metrics and loss
            test_metrics = {}
            for metric in metrics:
                test_metrics[metric.__class__.__name__] = float(metric.compute())
                metric.reset()

            test_metrics["Loss"] = loss_test / total_test
            test_metrics_history.append(test_metrics)

            print(f"Epoch {epoch + 1} / {num_epochs}")

            for metric, value in train_metrics.items():
                print(f"[TRAIN] {metric} = {value}")

            for metric, value in test_metrics.items():
                print(f"[TEST] {metric} = {value}")

            mx_metric = 0
            for i in range(len(test_metrics_history) - 1):
                mx_metric = max(mx_metric, test_metrics_history[i][checkpoint_metric])

            if test_metrics[checkpoint_metric] > mx_metric:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"{self.output_dir}/checkpoints/checkpoint_{epoch + 1}.pth")
                print(f"Saved to {self.output_dir}/checkpoints/checkpoint_{epoch + 1}.pth")

            fig, ax = plt.subplots((len(train_metrics.keys()) + 1) // 2, 2, figsize=(10, 3 * (len(train_metrics.keys()) + 1) // 2))
            ax = ax.flatten()

            for i, key in enumerate(train_metrics.keys()):
                train_values = list(map(lambda x: x[key], train_metrics_history))
                test_values = list(map(lambda x: x[key], test_metrics_history))
                coords = list(range(1, epoch + 2))

                ax[i].plot(coords, train_values, label=f'Train')
                ax[i].plot(coords, test_values, label=f'Test')

                ax[i].set_xlabel("Epoch")
                ax[i].set_ylabel(key)
                ax[i].set_title(key)
                ax[i].legend()
                ax[i].grid()

            plt.subplots_adjust(hspace=0.5)
            plt.savefig(f"{self.output_dir}/plots/{epoch + 1}.jpg")

            with open(f"{self.output_dir}/train_history.json", 'w') as f:
                json.dump(train_metrics_history, f)

            with open(f"{self.output_dir}/test_history.json", 'w') as f:
                json.dump(test_metrics_history, f)
