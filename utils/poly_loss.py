import torch

def poly1_ce(base_loss, logits, labels, epsilon=2.0):
    CE = base_loss(logits, labels)
    pt = torch.mean(torch.sum(torch.nn.functional.softmax(logits, -1) * labels, -1))
    return CE + epsilon * (1 - pt)