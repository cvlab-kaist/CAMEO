import torch

# loss function
def cross_entropy(prob, prob_gt):
    """Cross entropy loss for attention probabilities."""
    eps = 1e-8
    return - (prob_gt * (prob + eps).log()).sum(dim=-1).mean()

def kl_divergence(prob, prob_gt):
    """Kullback-Leibler divergence loss for attention probabilities."""
    return (prob_gt * (prob_gt.log() - prob.log())).sum(dim=-1).mean()

# def soft_argmax(prob, prob_gt, alpha=0.5):
#     """Soft argmax loss for attention probabilities.
#     Additional regularization to match entropy between two distributions"""
#     idxs = torch.arange(prob.shape[-1], device=prob.device).float()
#     argmax = (prob * idxs).sum(dim=-1)
#     argmax_gt = (prob_gt * idxs).sum(dim=-1)
#     argmax_loss = torch.nn.functional.l1_loss(argmax, argmax_gt)
#     eps = 1e-8
#     entropy_loss = ((prob_gt * (prob_gt + eps).log()).sum(dim=-1) - (prob * (prob + eps).log()).sum(dim=-1)).mean()
#     return argmax_loss + alpha * entropy_loss

ATTN_LOSS_FN = {
    "l1": torch.nn.functional.l1_loss,
    "mse": torch.nn.functional.mse_loss,
    "cross_entropy": cross_entropy,
    "kl_divergence": kl_divergence,
    # "soft_argmax": soft_argmax,
}