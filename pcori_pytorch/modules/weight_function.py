"""
Computes the weight function as defined in Equation 4 of:

    Segmental Recurrent Neural Networks. Kong et al. 2016
"""

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common import Params


class WeightFunction(torch.nn.Module):
    """This ``Module`` computes scores for segments. E.g. is the 'f' defined in
    Equation 4 of:

        Segmental Recurrent Neural Networks. Kong et al. 2016

    Scores are computed using 2 layer MLP which outputs a scalar value.
    """
    def __init__(self,
                 label_embedding_size: int,
                 duration_embedding_size: int,
                 segment_embedding_size: int,
                 hidden_size: int) -> None:
        super(WeightFunction, self).__init__()
        self.input_size = label_embedding_size + duration_embedding_size + segment_embedding_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self,
                embedded_label: torch.FloatTensor,
                embedded_duration: torch.FloatTensor,
                embedded_segment: torch.FloatTensor):
        input = torch.cat((embedded_label, embedded_duration, embedded_segment), -1)
        hidden = self.fc1(input)
        hidden = F.tanh(hidden)
        output = self.fc2(hidden)
        return output

    @classmethod
    def from_params(cls, params: Params):
        label_embedding_size = params.pop_int('label_embedding_size')
        duration_embedding_size = params.pop_int('duration_embedding_size')
        segment_embedding_size = params.pop_int('segment_embedding_size')
        hidden_size = params.pop_int('hidden_size')
        params.assert_empty(cls.__name__)
        return cls(label_embedding_size=label_embedding_size,
                   duration_embedding_size=duration_embedding_size,
                   segment_embedding_size=segment_embedding_size,
                   hidden_size=hidden_size)

