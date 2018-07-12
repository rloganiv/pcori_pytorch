# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class HierarchicalCRFTest(ModelTestCase):
    def setUp(self):
        super(HierarchicalCRFTest, self).setUp()
        self.set_up_model('tests/fixtures/experiment.json',
                          'tests/fixtures/test_sessions.jsonl')

    def test_model_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

