##@meta {desc: 'build and deployment for python projects', date: '2024-10-27'}


## Build system
#
#
# type of project
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy
INFO_TARGETS +=		appinfo
PY_DEP_POST_DEPS +=	lituievdeps traindeps
ADD_CLEAN +=		log
ADD_CLEAN_ALL +=	data
CLEAN_ALL_DEPS +=	cleanpaper


## Project
#
ENTRY = 		./harness.py
DATASETS ?=		mimic synthetic
DEFAULT_ARGS =		-c etc/model.conf --override sdoh_default.label=mimic,sdoh_lm_default.feature_hint_type=lm


## Includes
#
include ./zenbuild/main.mk



## Environment and utility targets
#
# build configuration
.PHONY:			appinfo
appinfo:
			@echo "app-resources-dir: $(RESOURCES_DIR)"

# install the additional features deps
.PHONY:			lituievdeps
lituievdeps:
			$(PIP_BIN) install lemma_tokenizer
			$(PIP_BIN) install --no-deps lib/en_sdoh_bow-0.0.2-py3-none-any.whl
			$(PYTHON_BIN) src/bin/lituievdeps-nltk-deps.py

# install dependencies that Conda can not
.PHONY:			traindeps
traindeps:
			$(PIP_BIN) install xformers==0.0.25.post1

## Clean targets
#
# remove all results and (temporary) data files--careful!
.PHONY:			vaporize
vaporize:		cleanall
			rm -fr corpus results
			git checkout corpus

# print a sample from the dataset used to SFT train
.PHONY:			sample
sample:
			$(ENTRY) $(DEFAULT_ARGS) sample

# print the trainer configuration
.PHONY:			trainer
trainer:
			$(ENTRY) $(DEFAULT_ARGS) trainer



# clean the paper contents
.PHONY:			cleanpaper
cleanpaper:
			make -C paper cleanall
