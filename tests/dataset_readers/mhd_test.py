# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from pcori_pytorch.dataset_readers import MHDDatasetReader


class TestMHDDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        # Read in the data
        reader = MHDDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/test_sessions.jsonl'))

        # Define our expectations
        instance0 = {'session_id': '1337',
                     'utterances': [['This', 'is', 'the', 'first', 'utterance', '.'],
                                    ['This', 'is', 'the', 'second', '.']],
                     'speaker_ids': ['speaker0', 'speaker1'],
                     'labels': ['label0', 'label1']}
        instance1 = {'session_id': '5150',
                     'utterances': [['Van', 'Halen'],
                                    ['Police', 'code', 'for', 'crazy', 'one', 'run', 'amok']],
                     'speaker_ids': ['speaker1', 'speaker0'],
                     'labels': ['label1', 'label0']}

        assert len(instances) == 2 # Ensure data has correct number of elements

        # Check first instance matches
        fields = instances[0].fields
        utterances = [[x.text for x in utterance.tokens] for utterance in fields['utterances'].field_list]

        assert utterances == instance0['utterances']
        speaker_ids = [x.text for x in fields['speaker_ids'].tokens]
        assert speaker_ids == instance0['speaker_ids']
        assert fields['labels'].labels == instance0['labels']

        # Check second instance matches
        fields = instances[1].fields
        utterances = [[x.text for x in utterance.tokens] for utterance in fields['utterances'].field_list]
        assert utterances == instance1['utterances']
        speaker_ids = [x.text for x in fields['speaker_ids'].tokens]
        assert speaker_ids == instance1['speaker_ids']
        assert fields['labels'].labels == instance1['labels']

