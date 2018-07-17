# pylint: disable=no-self-use,invalid-name
import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase

from pcori_pytorch.training import FuckingAccuracy


class TestFuckingAccuracy(AllenNlpTestCase):

    def test_output(self):
        fucking_accuracy = FuckingAccuracy()
        true = torch.LongTensor([[1, 2, 3], [4, 0, 0]])
        pred = torch.LongTensor([[1, 1, 1], [4, 0, 0]])
        mask = torch.FloatTensor([[1, 1, 1], [1, 0, 0]])
        fucking_accuracy(true, pred, mask)
        # Check metric is properly computed
        value = fucking_accuracy.get_metric(reset=True)
        assert value == 0.5
        # Check that resetting worked
        with pytest.raises(ZeroDivisionError):
            value = fucking_accuracy.get_metric()

