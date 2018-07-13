PCORI PyTorch
===

Neural sequence labeling models for the PCORI project.

To get started you will need to install the project dependencies in a Python3.6 environment:
```
pip install -r requirements.txt
```

An example model configuration is provided in the `experiments/for_collabs.json` file.
To train this model run:
```
allennlp train experiments/for_collabs.json \
    -s CKPT_DIR \
    --include-package pcori_pytorch
```

To output model predictions run:
```
allennlp predict \
    CKPT_DIR/model.tar.gz \
    DATA.jsonl \
    --include-package pcori_pytorch \
    --predictor hierarchical_crf_predictor
```

