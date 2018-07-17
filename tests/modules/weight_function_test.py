# pylint: disable=no-self-use,invalid-name
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase

from pcori_pytorch.modules import WeightFunction


class TestWeightFunction(AllenNlpTestCase):

    def test_output(self):
        batch_size = 4
        label_embedding_size = 8
        label_embedding = Variable(torch.randn(batch_size, label_embedding_size))

        duration_embedding_size = 8
        duration_embedding = Variable(torch.randn(batch_size, duration_embedding_size))

        segment_embedding_size = 16
        segment_embedding = Variable(torch.randn(batch_size, segment_embedding_size))

        hidden_size = 16
        weight_function = WeightFunction(label_embedding_size,
                                         duration_embedding_size,
                                         segment_embedding_size,
                                         hidden_size)
        output = weight_function(label_embedding, duration_embedding, segment_embedding)
        assert output.shape == (batch_size, 1)

