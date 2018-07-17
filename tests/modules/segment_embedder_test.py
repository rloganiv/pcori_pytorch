# pylint: disable=no-self-use,invalid-name
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase

from pcori_pytorch.modules import SegmentEmbedder
from pcori_pytorch.modules.segment_embedder import reverse


class TestReverse(AllenNlpTestCase):

    def test_reverse(self):
        input = torch.rand(1, 4, 1)
        input = Variable(input)
        reversed = reverse(input)
        output = reverse(reversed)
        assert torch.equal(input, output)

class TestSegmentEmbedder(AllenNlpTestCase):

    def test_upper_triangular(self):
        input = torch.rand(1, 4, 1)
        input = Variable(input)
        segment_embedder = SegmentEmbedder(1, 1)
        output = segment_embedder(input)
        for i in range(4):
            for j in range(i):
                if i != j:
                    # Check that lower triangular elements are all zero
                    is_zero = output[:,i,j,:] == 0
                    assert is_zero.all()

    def test_max_length(self):
        input = torch.rand(1, 4, 1)
        input = Variable(input)
        segment_embedder = SegmentEmbedder(1, 1, max_length=1)
        output = segment_embedder(input)
        for i in range(4):
            for j in range(i+1,4):
                # Check that above max_length elements are all zero
                is_zero = output[:,i,j,:] == 0
                assert is_zero.all()

    def test_mask(self):
        input = torch.rand(2, 4, 1)
        input = Variable(input)
        mask = torch.ByteTensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        mask = Variable(mask)
        segment_embedder = SegmentEmbedder(1, 1)
        output = segment_embedder(input, mask.float())
        # TODO: Actually test the output...

    def test_cuda(self):
        if not torch.cuda.is_available():
            return
        input = torch.rand(2, 4, 1).cuda()
        input = Variable(input)
        segment_embedder = SegmentEmbedder(1, 1).cuda()
        output = segment_embedder(input)

