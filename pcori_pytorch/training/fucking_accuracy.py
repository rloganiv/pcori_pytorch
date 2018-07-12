"""I should not need to write this..."""

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

@Metric.register("fucking_accuracy")
class FuckingAccuracy(Metric):
    def __init__(self):
        self._correct_count = 0.
        self._total_count = 0.

    def __call__(self, predictions, gold_labels, mask):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        correct = predictions.eq(gold_labels).float()
        if mask is not None:
            correct = correct * (1 - mask.float())
        self._correct_count += correct.sum()
        self._total_count += mask.sum()

    def get_metric(self, reset: bool = False):
        accuracy = float(self._correct_count) / float(self._total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0

