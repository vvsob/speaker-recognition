from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F


def extract_features(waveforms):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

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
    def __init__(self, model: nn.Module, train_ds, test_ds, batch_size, device=None):
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

    def fit(self, optimizer, loss, num_epochs=10, lr=0.001, metrics: list = None, schedulers: list = None):
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
                train_metrics[metric.__class__.__name__] = metric.compute()
                metric.reset()

            train_metrics["Loss"] = loss_train / total_train

            # evaluation
            self.model.eval()

            loss_test = 0
            total_test = 0

            with torch.no_grad():
                for inputs, labels in self.test_loader:
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
                test_metrics[metric.__class__.__name__] = metric.compute()
                metric.reset()

            test_metrics["Loss"] = loss_test / total_test

            print(f"Epoch {epoch + 1} / {num_epochs}")

            for metric, value in train_metrics.items():
                print(f"[TRAIN] {metric} = {value}")

            for metric, value in test_metrics.items():
                print(f"[TEST] {metric} = {value}")
