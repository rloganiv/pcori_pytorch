"""
Implementation of the Segmental RNN described in:

    Segmental Recurrent Neural Networks. Kong et al. 2016

adapted to make predictions on sequences of utterances instead of words.  E.g. An additional
encoding step must be performed to obtain a single vector representation of each utterance.
"""

from typing import Dict, Optional, List, Any

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Embedding, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util

from pcori_pytorch.modules import SegmentEmbedder, WeightFunction
from pcori_pytorch.training import FuckingAccuracy


@Model.register('hierarchical_srnn')
class HierarchicalSRNN(Model):
    """
    The ``HierarchicalSRNN`` encodes sequences of sequences of text and uses a segmental RNN model
    to segment the outer sequence and predict tags for each segment.

    Parameters
    ----------
    vocab: ``Vocabulary``
        A vocabulary, needed to compute the sizes for input/output projections.
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed input text, as well as labels.
    duration_embedder : ``Embedding``
        Used to embed segment durations.
    inner_encoder : ``Seq2VecEncoder``
        The encoder that will be used to encode inner sequences.
    outer_encoder : ``Seq2SeqEncoder``
        The encoder that will be used to encode outer sequences (e.g. the step performed before
        computing segment embeddings).
    segment_embedder : ``SegmentEmbedder``
        Used to embed segments.
    weight_function : ``WeightFunction``
        Used to compute scores for labeled segments.
    dropout : ``float``, optional (default=``None``)
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 duration_embedder: Embedding,
                 label_embedder: Embedding,
                 inner_encoder: Seq2VecEncoder,
                 outer_encoder: Seq2SeqEncoder,
                 segment_embedder: SegmentEmbedder,
                 weight_function: WeightFunction,
                 label_namespace: str = 'labels',
                 max_length: int = None,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(HierarchicalSRNN, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.max_length = max_length
        self.num_labels = self.vocab.get_vocab_size(label_namespace)
        self.text_field_embedder = text_field_embedder
        self.duration_embedder = duration_embedder
        self.label_embedder = label_embedder
        self.inner_encoder = inner_encoder
        self.outer_encoder = outer_encoder
        self.segment_embedder = segment_embedder
        self.weight_function = weight_function

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.metrics = {"accuracy": FuckingAccuracy()}

    def forward(self,
                utterances: Dict[str, torch.LongTensor],
                speakers: Dict[str, torch.LongTensor],
                labels: Dict[str, torch.LongTensor] = None,
                durations: Dict[str, torch.LongTensor] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Note: Currently ignores speaker data.

        Parameters
        ----------
        utterances : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()``, which should be passed directly to a
            ``TextFieldEmbedder``.
        speakers : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()``, which should be passed directly to a
            ``TextFieldEmbedder``.
        labels : ``Dict[str, torch.LongTensor]``, optional (default = ``None``)
            A torch tensor containing the observed segment labels.
        durations : ``Dict[str, torch.LongTensor]``, optional (default = ``None``)
            A torch tensor containing the observed segment label durations.
        metadata : ``List[Dict[str, Any]]``, option (default = ``None``)
            Metadata about the inputs (e.g. session_id, the original sequences of words, etc.).

        Returns
        -------
        An output dictionary consisting of:

        labels : ``List[List[str]]``
            The predicted segment labels (using the dynamic program given in Section 3.2).
        durations : ``List[List[int]]``
            The predicted segment durations (using the dynamic program given in Section 3.2).
        loss: ``torch.FloatTensor``, optional
            A scalar loss to be optimized. Only computed if ``labels`` and ``durations`` provided.
        """
        # Extract masks - note: there are two, the first is a word level mask (e.g. accounts for
        # padding the sequences of words within utterances), the second is an utterance level mask
        # (e.g. accounts for padding sequences of utterances), the third is a label mask (since in
        # the segmental model there is no longer a 1-to-1 correspondence between utterances and
        # labels).
        inner_mask = util.get_text_field_mask(utterances, 1)
        outer_mask = util.get_text_field_mask(utterances)
        outer_lengths = outer_mask.long().sum(-1).data
        label_mask = durations.gt(0) # The easy way!
        label_lengths = label_mask.long().sum(-1).data

        batch_size, n_utterances, n_words = inner_mask.shape

        # Embed
        embedded_utterances = self.text_field_embedder(utterances)
        if self.dropout:
            embedded_utterances = self.dropout(embedded_utterances)

        # Get inner encodings
        # TODO: Replace with TimeDistributed
        embedded_utterances = embedded_utterances.view(batch_size * n_utterances, n_words, -1)
        inner_mask = inner_mask.view(batch_size * n_utterances, n_words)
        inner_encoded = self.inner_encoder(embedded_utterances, inner_mask)
        if self.dropout:
            inner_encoded = self.dropout(inner_encoded)

        # Unflatten and get outer encodings
        inner_encoded = inner_encoded.view(batch_size, n_utterances, -1)
        outer_encoded = self.outer_encoder(inner_encoded, outer_mask)
        if self.dropout:
            outer_encoded = self.dropout(outer_encoded)

        # Get segment embeddings
        segment_embeddings = self.segment_embedder(outer_encoded, outer_mask.float())

        # Decode and compute partition function
        # TODO: Think of a better name for the outputs
        labels_, segments_, log_z = self.compute_map_and_log_z(segment_embeddings, outer_lengths)

        output_dict = {
            'labels': labels_,
            'segments': segments_
        }

        # Compute the unnormalized conditional probability of an observed sequence (if provided)
        if labels is not None:
            unnormalized_ll = Variable(torch.zeros(batch_size))
            if labels.is_cuda:
                unnormalized_ll = unnormalized_ll.cuda()
            for k in range(batch_size):
                s = 0
                log_weight = 0.0
                for j in range(label_lengths[k]):
                    t = s + durations.data[k, j] - 1
                    embedded_label = self.label_embedder(labels[k, j])
                    embedded_duration = self.duration_embedder(durations[k, j])
                    embedded_segment = segment_embeddings[k, s, t].view(1, -1)
                    log_weight += self.weight_function(embedded_label, embedded_duration,
                                                       embedded_segment)
                unnormalized_ll[k] = log_weight
            loss = log_z.sum() - unnormalized_ll.sum()
            output_dict['loss'] = loss

        return output_dict

    # TODO: Figure out the proper type hints.
    def compute_map_and_log_z(self,
                              segment_embeddings: torch.Tensor,
                              outer_lengths: torch.Tensor):
        """
        Computes the MAP segmentation and log partition function.

        These are calculated simultaneously to avoid redundant computation.

        Parameters
        ----------
        segment_embeddings : ``torch.FloatTensor``
            Segment embeddings output by a ``SegmentEmbedder``.
        outer_lengths : ``torch.LongTensor``
            Stores the lengths of each session in the current batch.

        Returns
        -------
        map : List[Any]
            The maximum a priori segmentation of the input sequence.
        log_z : torch.Tensor
            The partition function evaluated for each input.
        """
        batch_size = outer_lengths.shape[0]

        log_z = Variable(torch.zeros(batch_size))
        if segment_embeddings.is_cuda:
            log_z = log_z.cuda()

        # WARNING: The indexing in this process is really difficult to keep track of, note that
        # alpha[t+1] is for the t'th element in the input sequence (so the length of alpha is one
        # greater thant the number of inputs), all durations are positive, but confusingly a length
        # 1 segment corresponds to a segment embedding where s == t.
        for k in range(batch_size):

            log_alphas = Variable(torch.zeros(outer_lengths[k] + 1))
            if segment_embeddings.is_cuda:
                log_alphas = log_alphas.cuda()

            for t in range(outer_lengths[k]):
                l = max(0, t - self.max_length)
                alpha_t = 0.0
                for s in range(l, t + 1):

                    alpha_s = log_alphas.exp()[s] # Dumb trick - shouldn't need to exp everything

                    duration = Variable(torch.LongTensor([t - s + 1]))
                    if segment_embeddings.is_cuda:
                        duration = duration.cuda()

                    embedded_duration = self.duration_embedder(duration).view(1, -1)
                    inner_sum = 0.0
                    for l in range(self.num_labels):

                        label = Variable(torch.LongTensor([l]))
                        if segment_embeddings.is_cuda:
                            label = label.cuda()

                        embedded_label = self.label_embedder(label).view(1, -1)
                        embedded_segment = segment_embeddings[k, s, t].view(1, -1)
                        weight = self.weight_function(embedded_label,
                                                      embedded_duration,
                                                      embedded_segment)
                        inner_sum += torch.exp(weight)
                    alpha_t += alpha_s * inner_sum
                log_alphas[t + 1] = torch.log(alpha_t)
            log_z[k] = log_alphas[-1]

        return None, None, log_z

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'HierarchicalCRF':
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        duration_embedder = Embedding.from_params(None, params.pop('duration_embedder'))
        label_embedder = Embedding.from_params(vocab, params.pop('label_embedder'))
        inner_encoder = Seq2VecEncoder.from_params(params.pop('inner_encoder'))
        outer_encoder = Seq2SeqEncoder.from_params(params.pop('outer_encoder'))
        segment_embedder = SegmentEmbedder.from_params(params.pop('segment_embedder'))
        weight_function = WeightFunction.from_params(params.pop('weight_function'))
        label_namespace = params.pop('label_namespace', 'labels')
        max_length = params.pop_int('max_length', None)
        dropout = params.pop_float('dropout', None)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   duration_embedder=duration_embedder,
                   label_embedder=label_embedder,
                   inner_encoder=inner_encoder,
                   outer_encoder=outer_encoder,
                   segment_embedder=segment_embedder,
                   weight_function=weight_function,
                   label_namespace=label_namespace,
                   max_length=max_length,
                   dropout=dropout,
                   initializer=initializer,
                   regularizer=regularizer)

