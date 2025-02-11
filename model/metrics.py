import torch
from torcheval.metrics.functional import multiclass_f1_score


class MulticlassF1Score:
    def __init__(self):
        self.inputs = torch.Tensor([])
        self.targets = torch.Tensor([])

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.argmax(dim=1)

        self.inputs = torch.cat((self.inputs.to(inputs.device), inputs))
        self.targets = torch.cat((self.targets.to(targets.device), targets))

    def reset(self):
        self.inputs = torch.Tensor([])
        self.targets = torch.Tensor([])

    def compute(self):
        num_classes = len(set(self.inputs).union(set(self.targets)))
        return multiclass_f1_score(self.inputs, self.targets, num_classes=num_classes)

