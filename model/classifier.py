from model.predictor import Predictor
import torch


class Classifier:
    def __init__(self, model_path="best_checkpoint.pth"):
        self.predictor = Predictor(6, model_path)

    def get_probabilities(self, array: torch.Tensor, sampling_rate: int, window_length: int = 5) -> torch.Tensor:
        n_samples = int(array.shape[1])
        window = window_length * sampling_rate

        predictions: list[torch.Tensor] = []
        for start in range(0, n_samples, window):
            end = min(start + window, n_samples)
            pred = self.predictor.predict(array[:, start:end], sampling_rate)
            pred *= (end - start) / window

            predictions.append(pred)

        result = sum(predictions)
        result /= sum(result)
        return result

    def get_id_probabilities(self, array, sampling_rate, window_length=5):
        probabilities = self.get_probabilities(array, sampling_rate, window_length)
        return int(torch.abs(probabilities).argmax()), probabilities
