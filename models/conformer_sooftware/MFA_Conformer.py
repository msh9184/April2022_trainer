import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from typing import Tuple
from utils import PreEmphasis
from .encoder import ConformerEncoder
from .modules import Linear
from .specaugment import SpecAugment

class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: x
        - **x** (batch, time, dim): Tensor containing input vector

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 256,
            num_encoder_layers: int = 6,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 8,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 15,
            half_step_residual: bool = True,
            num_out: int = 192,
            **kwargs
    ) -> None:
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        output_dim = encoder_dim*num_encoder_layers
        self.specaug = SpecAugment()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        self.attention = nn.Sequential(
            nn.Conv1d(output_dim*3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn1 = nn.BatchNorm1d(output_dim*2)
        self.fc  = nn.Linear(output_dim*2, num_out)
        self.bn2 = nn.BatchNorm1d(num_out)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, x: Tensor, aug=False) -> Tuple[Tensor, bool]:

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if aug == True:
                    x = self.specaug(x)
        # ConformerBlocks
        x = self.encoder(x)
        # Context dependent ASP
        t = x.size()[-1]
        global_x = torch.cat((x,torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu, sg), dim=1)
        # BN -> FC: embedding
        x = self.bn1(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x

def MainModel(num_mels=80, num_out=192, **kwargs):
    model = Conformer(input_dim=num_mels, encoder_dim=256, num_encoder_layers=6, num_attention_heads=4, feed_forward_expansion_factor=8, conv_kernel_size=15, num_out=num_out, **kwargs)
    return model
