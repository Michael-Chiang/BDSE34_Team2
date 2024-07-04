import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OrdinalCrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,)
        """
        batch_size, num_classes = outputs.size()

        # Create an ordinal label matrix
        ordinal_labels = torch.zeros(
            batch_size, num_classes).to(outputs.device)
        for i in range(batch_size):
            ordinal_labels[i, :targets[i] + 1] = 1
        print(ordinal_labels)
        # Apply sigmoid to the outputs
        sigmoid_outputs = torch.sigmoid(outputs)

        # Calculate ordinal cross entropy loss
        loss = F.binary_cross_entropy(sigmoid_outputs, ordinal_labels)

        return loss
