# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from pcori_pytorch.data.dataset_readers import SegmentedMHDDatasetReader


class TestSegmentedMHDDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        # Read in the data
        reader = SegmentedMHDDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/test_segments.jsonl'))

        # Define our expectations
        instance0 = {'session_id': '1337',
                     'utterances': [['First', 'utterance'],
                                    ['Second', 'utterance'],
                                    ['Third', 'utterance', 'with', 'a', 'different', 'label']],
                     'speakers': ['speaker0', 'speaker1', 'speaker0'],
                     'labels': ['label0', 'label1'],
                     'durations': [2, 1] }

        assert len(instances) == 2 # Ensure data has correct number of elements

        # Check first instance matches
        fields = instances[0].fields
        utterances = [[x.text for x in utterance.tokens] for utterance in fields['utterances'].field_list]

        assert utterances == instance0['utterances']
        speakers = [x.text for x in fields['speakers'].tokens]
        assert speakers == instance0['speakers']
        assert fields['labels'].labels == instance0['labels']
        print(fields['durations'])
        assert fields['durations'].int_list == instance0['durations']

