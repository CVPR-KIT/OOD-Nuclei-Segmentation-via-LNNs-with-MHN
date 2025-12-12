from torch import nn
from vector_quantize_pytorch import ResidualVQ
    

class ResidualQuantization(nn.Module):
    def __init__(self, input_dim: int, num_quantizers: int, codebook_size: int = 512, accept_image_fmap=True,
                 shared_codebook: bool = True, implicit_neural_codebook: bool = False, 
                 stochastic_sample_codes: bool = False, sample_codebook_temp: float = 0.1):
        """
        Residual Quantization Module.

        Args:
            input_dim (int): Dimension of the input embeddings.
            num_quantizers (int): Number of quantizers (Residual VQ layers).
            codebook_size (int, optional): Number of vectors in each codebook. Defaults to 512.
            shared_codebook (bool, optional): Whether to share the codebook across quantizers. Defaults to True.
            implicit_neural_codebook (bool, optional): If True, uses an implicit neural codebook. Defaults to False.
            stochastic_sample_codes (bool, optional): If True, samples codes stochastically. Defaults to False.
            sample_codebook_temp (float, optional): Temperature for stochastic sampling. Defaults to 0.1.
        """
        super(ResidualQuantization, self).__init__()
        self.quantize = ResidualVQ(dim=input_dim, num_quantizers=num_quantizers, codebook_size=codebook_size, accept_image_fmap=accept_image_fmap,
                                   shared_codebook=shared_codebook, implicit_neural_codebook=implicit_neural_codebook, 
                                   stochastic_sample_codes=stochastic_sample_codes, sample_codebook_temp=sample_codebook_temp)

    def forward(self, x):
        """
        Forward pass for Residual Quantization.

        Args:
            x (Tensor): Input embeddings of shape [batch_size * H * W, embedding_dim].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Quantized embeddings, indices, and commitment loss.
        """
        quantized, indices, commit_loss = self.quantize(x)
        return quantized, indices, commit_loss
    
    @property
    def codebook(self):
        return self.quantize.codebook