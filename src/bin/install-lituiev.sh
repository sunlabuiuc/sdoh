#!/bin/bash

# Install the Lituiev et al. (2023) SDoH hybrid model with the Python
# environment it needs

PROG=$(basename $0)
SDOH_DIR=social-determinants-of-health-clbp
REPO_URL=https://github.com/BCHSI/${SDOH_DIR}
CONDA_DIR=$(pwd)/env
PYTHON_BIN=${CONDA_DIR}/bin/python
ENV_FILE=environment.yml
ENV_CONF="\
name: sdoh
channels:
  - defaults
dependencies:
  - python=3.9
  - scikit-learn==1.0.1
  - pip
  - pip:
    - spacy==3.2.6
    - numpy==1.26.4
    - lemma_tokenizer"

function prhead() {
    echo "--------------------${1}:"
}

function bail() {
    msg=$1 ; shift
    echo "$PROG: error: $msg"
    exit 1
}

function assert_success() {
    ret=$1 ; shift
    if [ $ret -ne 0 ] ; then
	bail "last command failed"
    fi
}

function clean() {
    prhead "removing any old artifcats"
    rm -fr ${CONDA_DIR} ${ENV_FILE}
}    

function cleanall() {
    clean
    rm -fr ${SDOH_DIR} 
}

function install_env() {
    printf "${ENV_CONF}\n" > ${ENV_FILE}
    [ ! -f ${ENV_FILE} ] && assert_success 1
    if [ ! -d ${SDOH_DIR} ] ; then
	git clone ${REPO_URL}
	assert_success $?
    fi
    conda env create -f environment.yml --prefix=${CONDA_DIR}
    assert_success $?
    ${PYTHON_BIN} -m spacy download en_core_web_md
    assert_success $?
}

function install_sdoh() {
    ( cd ${SDOH_DIR}/model-hybrid-bow/package/en_sdoh_bow-0.0.2 ;
      ${PYTHON_BIN} setup.py clean ;
      ${PYTHON_BIN} setup.py install )
    assert_success $?
}

function install_nltk_deps() {
    ${PYTHON_BIN} -c "
import nltk
for i in 'averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords'.split():
  nltk.download(i)"
    assert_success $?
}

function install() {
    prhead "installing sdoh enviornment"
    install_env
    install_sdoh
    install_nltk_deps
}

function test_model() {
    prhead "test model"
    ${PYTHON_BIN} src/bin/test-lituiev-model.sh
    assert_success $?
    echo "test model...ok"
}

function main() {
    cleanall
    install
    test_model
}

main
