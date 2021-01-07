"""
"curry-net" (c) by Ignacio Slater M.
"curry-net" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, LSTM, Linear, functional


class RecurrentNetwork(torch.nn.Module):
    """
    Implementation of a basic recurrent neural network with autoregression for language modeling.
    """
    __classifier: Linear
    __lstm: LSTM

    def __init__(self, vocab_size: int, output_size: int, embedding_dim: int, hidden_dim: int,
                 n_layers: int, pad_idx: int, dropout: float = 0.5):
        """
        Creates a new instance of the network following the GRU architecture and using an embedding
        layer.

        Args:
            vocab_size:
                the number of elements on the vocabulary.
            output_size:
                 the number of categories the network classifies.
            embedding_dim:
                the dimensions of the embeding layer.
            hidden_dim:
                the dimension of the initial hidden state.
            n_layers:
                the number of layers in the network.
            pad_idx:
                an index to represent the padding.
        """
        super(RecurrentNetwork, self).__init__()
        # embedding = Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.__lstm = LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.__classifier = Linear(hidden_dim, output_size)

    def forward(self, nn_input: Tensor, h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        out, h = self.__lstm(nn_input, h_0)
        out = self.__classifier(functional.relu(out[:, -1]))
        return out, h
