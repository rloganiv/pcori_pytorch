"""
Computes the segment embeddings as described in Section 3.1 of:

    Segmental Recurrent Neural Networks. Kong et al. 2016
"""

import torch
from overrides import overrides
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.modules import Seq2SeqEncoder


def reverse(inputs: Variable,
            dim: int = 1) -> Variable:
    """Reverses a sequence.

    This is needed for computing the backward segment embeddings, since negative strides are not
    allowed when slicing a ``torch.Tensor``. The main reason this function is not a one liner is
    that the reversed sequence indices need to be converted to a ``Variable``.

    Parameters
    ----------
    inputs: ``torch.autograd.Variable``
        The input tensor to reverse.
    dim: ``int``
        The dimension to reverse along. Default: 1.
    """
    _, seq_length, _ = inputs.shape
    idx = list(reversed(range(seq_length)))
    idx = inputs.data.new(idx).long()
    idx = Variable(idx)
    # WARNING: May need to use ``.contiguous``
    return inputs.index_select(dim, idx)


class SegmentEmbedder(torch.nn.Module):
    """This ``Module`` computes segment embeddings.

    Parameters
    ----------
    input_size : ``int``
        Size of inputs to the embedder.
    hidden_size : ``int``
        Size of embedder outputs.
    dropout: ``float``
        Dropout rate.
    max_length : ``int``
        The maximum allowed length of a span. If not allowed then spans are allowed to cover the
        entire sequence.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 max_length: int = None) -> None:
        super(SegmentEmbedder, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._max_length = max_length
        self._forward_lstm = torch.nn.LSTM(input_size, hidden_size, dropout=dropout,
                                           batch_first=True)
        self._backward_lstm = torch.nn.LSTM(input_size, hidden_size, dropout=dropout,
                                            batch_first=True)

    def reset_parameters(self):
        self._forward_lstm.reset_parameters()
        self._backward_lstm.reset_parameters()

    def forward(self,
                inputs: torch.FloatTensor,
                mask: torch.FloatTensor = None) -> torch.FloatTensor:
        """Encodes all possible segments of a sequence.

        inputs : ``torch.FloatTensor``, required
            A tensor of shape (batch_size, seq_length, input_size) containing the embeddings of the
            sequence elements.
        mask : ``torch.FloatTensor``, optional (default = ``None``)
            A torch tensor
        """
        use_cuda = inputs.is_cuda
        batch_size, seq_length, _ = inputs.shape

        if self._max_length is None:
            max_length = seq_length
        else:
            max_length = self._max_length

        # WARNING: Some stupid shit will happen with cuda, I guarantee it...

        # ``forward[i,j,k,:]`` will hold the forward embedding for the segment spanning tokens j
        # though k for the i'th sequence in the batch
        forward = torch.zeros(batch_size, seq_length, seq_length, self._hidden_size)
        forward = Variable(forward).cuda() if use_cuda else Variable(forward)
        for i in range(seq_length):
            j = min(seq_length, i + max_length)
            row, _ = self._forward_lstm(inputs[:,i:j,:])
            forward[:,i,i:j,:] = row

        # ``backward[i,j,k,:]`` will hold the backward embedding for the segment spanning tokens j
        # though k for the i'th sequence in the batch
        backward = torch.zeros(batch_size, seq_length, seq_length, self._hidden_size)
        backward = Variable(backward).cuda() if use_cuda else Variable(backward)
        reversed_inputs = reverse(inputs) # Need to reverse to feed into backwards LSTM
        # The idea is to fill in the reverse matrix from the bottom up then transpose to get its
        # entries to line up with forward. (See picture in notebook).
        for i in range(seq_length):
            j = min(seq_length, i + max_length)
            reversed_row, _ = self._backward_lstm(reversed_inputs[:,i:j,:])
            row = reverse(reversed_row)
            backward[:,(seq_length-i-1),(seq_length-j):(seq_length-i),:] = row
        backward = backward.transpose(1, 2)

        # Concatenate and mask
        output = torch.cat((forward, backward), -1)
        if mask is not None:
            # Segment mask converts the 3d mask over sequence elements to a 4d mask over segments
            segment_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            output = output * segment_mask.unsqueeze(-1)

        return output

    @classmethod
    def from_params(cls, params: Params):
         input_size = params.pop_int('input_size')
         hidden_size = params.pop_int('hidden_size')
         dropout = params.pop_float('dropout', 0.0)
         max_length = params.pop_int('max_length', None)
         params.assert_empty(cls.__name__)
         return cls(input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    max_length=max_length)

