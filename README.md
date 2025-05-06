# Social Determinants of Health Prediction

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.11][python311-badge]][python311-link]

A model that predicts Social Determinants of Health.


## Documentation

See the [full documentation](https://sunlabuiuc.github.io/sdoh/index.html).
The [API reference](https://sunlabuiuc.github.io/sdoh/api.html) is also
available.


## Obtaining

The library can be installed with pip from the [pypi] repository:
```bash
pip3 install zensols.sdoh
```

## Setup

1. Create a new Conda environment: `conda env create -f src/python/environment-lock.yml`
1. Activate it: `activate sdoh`
1. Download the [corpus]
1. Rename the corpus:
   ```bash
   cd download
   mv mimic-iii-clinical-care-database-1.0.1.zip annotation-dataset-of-social-determinants-of-health-from-mimic-iii-clinical-care-database-1.0.1.zip
   ```

## Installing Lituiev et al. (2023) SDoH Models

Optionally install the Lituiev et al. (2023) SDoH Models.


### spaCy hybrid model

1. Clone the spaCy package: `git clone https://github.com/BCHSI/social-determinants-of-health-clbp`
1. Change working directory: `cd social-determinants-of-health-clbp/model-hybrid-bow/package/en_sdoh_bow-0.0.2`
1. Build the wheel: `pip install wheel ; python setup.py bdist_wheel`
1. Install the wheel: `pip install --no-deps dist/en_sdoh_bow-0.0.2-py3-none-any.whl`
1. Install dependencies: `pip install lemma_tokenizer`
1. Install NLTK's dependencies: `pip install nltk ; python -c "import nltk ; nltk.download('stopwords') ; nltk.download('averaged_perceptron_tagger') ; nltk.download('wordnet')"`


### spaCy CNN Model

1. Clone the model repo: `git clone https://github.com/BCHSI/social-determinants-of-health-clbp`
1. Change working directory: `cd social-determinants-of-health-clbp/model-cnn-ner/packages/en_sdoh_cnn_ner_cui-0.0.0`
1. Build the wheel: `pip install wheel ; python setup.py bdist_wheel`
1. Install the wheel: `( cd dist ; pip install en_sdoh_cnn_ner_cui-0.0.0-py3-none-any.whl )`


### Convert the spaCy model to PyTorch

The HuggingFace 2023 model is for an old version of spaCy (3.2).  This converts
it to a HuggingFace model from a spaCy [source model].  It needs a previous
version of pip, so install an old version, install the spaCy model, then
restore pip.  Then the PyTorch model is converted to `sdoh-roberta-base`.

1. Remember the old version: `OLDVER=$(pip --version | awk '{print $2'})`
1. Compatible pip version for package: `pip install --upgrade pip==23`.
1. Install Git Large File System: `brew install git-lfs`
1. Download the model: `git clone https://huggingface.co/dlituiev/en_sdoh_roberta_cui`
1. Install it: `pip install --no-deps en_sdoh_roberta_cui/en_sdoh_roberta_cui-any-py3-none-any.whl`
1. Install dependencies: `pip install spacy-transformers`
1. The conversion script needs an older HF package: `pip install transformers==4.26`
1. Convert to the PyTorch model: `src/bin/topytorch.py`
1. Revert the pip version: `pip install --upgrade pip==${OLDVER}`
1. Cleanup: `rm -rf en_sdoh_roberta_cui`


## Training

1. Set the path to the configuration file:
   ```bash
   export SDOHRC=etc/model.conf
   ```
1. All and testing commands are given with the `harness` script. See the
   command line help: `./harness -h`
1. Run the fewshot LLM tests:
   ```bash
   for i in mimic synthetic mimthetic ; do
       ./harness fewshot $i
   done
   ```
1. Supervise-fine tune the LLM models, then test
   ```bash
   for i in mimic synthetic mimthetic ; do
       ./harness train $i
   done
   for i in mimic synthetic mimthetic ; do
       ./harness test $i
   done
   ```
1. Train and ablation test the traditional deep learning models:
   ```bash
   ./harness binaryabl
   for i in mimic synthetic mimthetic ; do
       ./harness multilabelabl
   done
   ```


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## Community

Please star this repository and let me know how and where you use this API.
Contributions as pull requests, feedback and any input is welcome.


## License

[MIT License](LICENSE.md)

Copyright (c) 2024 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.sdoh/
[pypi-link]: https://pypi.python.org/pypi/zensols.sdoh
[pypi-badge]: https://img.shields.io/pypi/v/zensols.sdoh.svg

[source model]: https://github.com/BCHSI/social-determinants-of-health-clbp
