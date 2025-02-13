import torch


def calc_eer_for_binary_classification(targets_scores: torch.Tensor, imposter_scores: torch.Tensor, iters: int = 100) -> float:
    """
    Computes the Equal Error Rate (EER) for a binary classification system.

    The EER is the point where the False Acceptance Rate (FAR) and the False Rejection Rate (FRR) are equal.

    :param targets_scores: torch.Tensor, scores assigned to genuine (target) samples.
    :param imposter_scores: torch.Tensor, scores assigned to imposter samples.
    :param iters: int, number of threshold iterations to evaluate EER.
    :return: float, the computed Equal Error Rate (EER).

    Errors:
    - Returns NaN if either targets_scores or imposter_scores is empty.
    """

    if len(targets_scores) == 0 or len(imposter_scores) == 0:
        return float('nan')
    min_score = torch.min(targets_scores.min(), imposter_scores.min())
    max_score = torch.max(targets_scores.max(), imposter_scores.max())

    n_tars = targets_scores.numel()
    n_imps = imposter_scores.numel()

    dists = torch.linspace(min_score, max_score, iters)
    fars = torch.zeros(iters, dtype=torch.float32)
    frrs = torch.zeros(iters, dtype=torch.float32)

    min_diff = float('inf')
    eer = 0

    for i, dist in enumerate(dists):
        far = (imposter_scores >= dist).sum().float() / n_imps
        frr = (targets_scores <= dist).sum().float() / n_tars
        fars[i] = far
        frrs[i] = frr

        diff = torch.abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2

    return eer.item()


def calc_eer(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """
    Computes the Equal Error Rate (EER) for a multi-class classification problem using PyTorch tensors.

    The function calculates EER for each class separately by treating it as a binary classification problem.
    It then returns a tensor containing the EER for each class.

    :param target: torch.Tensor, a 1D tensor (N,) containing the true class labels.
    :param predicted: torch.Tensor, a 2D tensor (N, C), where N is the number of samples and C is the number of classes.
                      Each row contains the predicted scores or probabilities for each class.
    """

    num_classes = predicted.shape[1]
    all_indices = torch.arange(num_classes)
    eer = torch.zeros((num_classes,))
    for cl in range(num_classes):
        fixed_indices = torch.where(target == cl)[0]
        other_indices = torch.where(target != cl)[0]
        fixed = predicted[fixed_indices, cl]
        other = predicted[other_indices, cl]
        eer[cl] = calc_eer_for_binary_classification(fixed, other)

    return eer