"""
Simple label propagation model.

Utterances are encoded into sequence of vectors using a ``Seq2SeqEncoder``.
Attention is paid to the output of this encoder - the idea being that signal words in the input
will be attended to the most.
Next a dense layer with sigmoid activation will be used to determine P(z | h), the probability that
the current input is relevant - if it is not then the previous prediction will be used (really it will
be an expectation).
Then projection to the output.
"""

from typing import Any, Dict, List, Optional

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import 
from overrides import overrides
from torch.nn.parameter import Parameter


@Model.register('label_prop'):
class LabelProp(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LabelProp, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size('labels')
        self.encoder = encoder
        self.attention = attention
        self.z_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(), 1))
        self.projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                       self.num_tags))

        self.W_attn = Parameter(torch.Tensor(self.encoder.get_output_dim()))
        self.W_logit0 = Parameter(torch.Tensor(self.num_tags)) # TODO: expand dims ???

        initializer(self)

    @overrides
    def forward(self,
                utterances: Dict[str, torch.LongTensor],
                speakers: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """Documentation"""
        # Extract masks - note: there are two, the first is a word level mask (e.g. accounts for
        # padding the sequences of words within utterances), the second is an utterance level mask
        # (e.g. accounts for padding sequences of utterances).
        inner_mask = util.get_text_field_mask(utterances, 1)
        outer_mask = util.get_text_field_mask(utterances)
        batch_size, n_utterances, n_words = inner_mask.shape

        # Lookup / compute embeddings
        embedded_utterances = self.text_field_embedder(utterances)

        # Feed into encoder
        embedded_utterances = embedded_utterances.view(batch_size * n_utterances, n_words, -1)
        inner_mask = inner_mask.view(batch_size * n_utterances, n_words)
        inner_encoded = self.inner_encoder(embedded_utterances, inner_mask)

