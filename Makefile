ifeq (, $(shell which python))
  $(error "Python not found in PATH - have you installed it?")
endif

PYTHON_REQUIRED_VERSION=3
PYTHON_VERSION=$(shell python -V 2>&1 | cut -d ' ' -f 2 | cut -d . -f 1)
PYTHON_FULL_VERSION=$(shell python -V 2>&1)

ifneq ($(PYTHON_REQUIRED_VERSION), $(PYTHON_VERSION))
  $(error "You are running $(PYTHON_FULL_VERSION). This repository requires Python $(PYTHON_REQUIRED_VERSION).")
endif

# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
HYDRA_FLAGS := -m
USE_WANDB := True
SEED := 0

## DENSITY : pass multiple densities via commandline
DENSITY := 0.2

.PHONY: help docs

.DEFAULT: help

## install: install all dependencies
install:
	pip install -r requirements.txt --upgrade
	pip install -e .

## docs: build documentation
docs.%:
	$(MAKE) -C docs_src $*

help : Makefile makefiles/*.mk
    ifeq ($(UNAME_S),Linux)
		@sed -ns -e '$$a\\' -e 's/^##//p' $^
    endif
    ifeq ($(UNAME_S),Darwin)
        ifneq (, $(shell which gsed))
			@gsed -sn -e 's/^##//p' -e '$$a\\' $^
        else
			@sed -n 's/^##//p' $^
        endif
    endif

include makefiles/*.mk

all: cifar10 cifar100 cifar10_tune vis