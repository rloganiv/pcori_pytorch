from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('hierarchical_srnn_predictor')
class HierarchicalSRNNPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        session_id = json_dict['session_id']
        utterances = json_dict['utterances']
        speakers = json_dict['speakers']
        labels = json_dict['labels']
        durations = json_dict['durations']
        instance = self._dataset_reader.text_to_instance(session_id=session_id,
                                                         utterances=utterances,
                                                         speakers=speakers,
                                                         labels=labels,
                                                         durations=durations)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        return instance, {'all_labels': all_labels}

