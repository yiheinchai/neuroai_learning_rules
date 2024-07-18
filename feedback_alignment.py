import torch


class FeedbackAlignment(torch.autograd.Function):
    """
    Errors are propagated through random feedback weights.
    """

    @staticmethod
    def forward(context, input, weight):
        """
        Forward pass method for the layer. Computes the output of the layer and
        stores variables needed for the backward pass.

        Arguments:
        - context (torch context): context in which variables can be stored for
          the backward pass.
        - input (torch tensor): input to the layer.
        - weight (torch tensor): layer weights.

        Returns:
        - output (torch tensor): layer output.
        """

        # compute the output for the layer (linear layer)
        output = input.mm(weight.t())

        # store variables in the context for the backward pass
        context.save_for_backward(input, weight, output)

        return output

    @staticmethod
    def backward(context):
        input, weight, output = context.saved_tensors

        output: torch.Tensor = output

        RANDOM_WEIGHTS_MEAN = 1
        RANDOM_WEIGHTS_STD = 1
        # TODO: Not sure what the best weights and std for feedback alignment is

        rand_weights = torch.normal(
            mean=RANDOM_WEIGHTS_MEAN, std=RANDOM_WEIGHTS_STD, size=weight.shape
        )

        grad_input = output.mm(rand_weights)

        return grad_input


class FeedbackAlignmentNetwork(torch.nn.Module):
    pass
