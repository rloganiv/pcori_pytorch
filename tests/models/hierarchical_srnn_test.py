# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class HierarchicalSRNNTest(ModelTestCase):
    def setUp(self):
        super(HierarchicalSRNNTest, self).setUp()
        self.set_up_model('tests/fixtures/hierarchical_srnn_experiment.json',
                          'tests/fixtures/test_segments.jsonl')

    def test_model_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

