from model.predictor import Predictor
import torch


class Classifier:
    def __init__(self, model_path="best_checkpoint.pth"):
        self.predictor = Predictor(6, model_path)

    def get_probabilities(self, array, sampling_rate, window_length=5):
        n_samples = int(array.shape[1])
        window = window_length * sampling_rate
        predictions = \
            [self.predictor.predict(array[:, start:min(start + window, n_samples)], sampling_rate) * ((min(start + window, n_samples) - start) / window) for start in
             range(0, n_samples, window)]
        result = sum(predictions)
        result /= sum(result)
        # result = "\n".join(str(predictions) for predictions in predictions)
        return result

    def get_id_probabilities(self, array, sampling_rate, window_length=5):
        probabilities = self.get_probabilities(array, sampling_rate, window_length)
        return str(int(torch.abs(probabilities).argmax())), probabilities