"""
Defines a ``DatasetReader`` for the MHD data.

Based on the tutorial at:
    https://github.com/allenai/allennlp-as-a-library-example/
"""

from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


@DatasetReader.register('mhd')
class MHDDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sessions from the MHD study. Each line is expected to have
    the following format:

        {
            'session_id': '...',
            'utterances': [
                ['Tokens', 'in', 'first', 'utterance', ...],
                ...,
                ['Tokens', 'in', 'last', 'utterance', ...]
            ],
            'speakers': ['...'],
            'labels': ['...']
        }

    The output of ``read`` is a list of ``Instance``s with the fields:
        TO BE DETERMINED

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``. If this is ``True``, training will start
        sooner but take longer per batch. Use if dataset is too large to fit in
        memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer used to split the utterances into tokens. Defaults to
        ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{'tokens': SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super(MHDDatasetReader, self).__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)
            for line in data_file:
                line = line.strip('\n')
                if not line:
                    continue
                session_json = json.loads(line)
                session_id = session_json['session_id']
                utterances = session_json['utterances']
                speakers = session_json['speakers']
                labels = session_json['labels']
                yield self.text_to_instance(session_id, utterances, speakers, labels)

    @overrides
    def text_to_instance(self,
                         session_id: str,
                         utterances: List[List[str]],
                         speakers: List[str],
                         labels: List[str] = None) -> Instance: # type: ignore
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # Since each session consists of a sequence of utterances, which themselves are sequences
        # of words, we need to use a ``ListField`` to properly handle the nested structure.
        tokenized_utterances = [[Token(word) for word in utterance] for utterance in utterances]
        sequence = ListField([TextField(x, self._token_indexers) for x in tokenized_utterances])
        fields['utterances'] = sequence
        # Speaker ids however are just a ``TextField`` since there is only one per utterance.
        # The tokens are constructed manually instead of fed into the tokenizer so that the text
        # isn't altered.
        # TODO: Since self._token_indexers is used to index both the words and speaker ids, the
        # speaker id's will be part of the Vocabulary - this may cause issues if the labels appear
        # in the text. Perhaps use a seperate indexer?
        tokenized_speakers = [Token(speaker) for speaker in speakers]
        fields['speakers'] = TextField(tokenized_speakers, self._token_indexers)
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sequence)
        # Add metadata to help with debugging / visualization.
        fields['metadata'] = MetadataField({
            'session_id': session_id,
            'utterances_text': [[x.text for x in utterance] for utterance in tokenized_utterances],
            'speakers_text': [x.text for x in tokenized_speakers]
        })
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'MHDDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers)

