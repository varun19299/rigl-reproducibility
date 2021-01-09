# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
HYDRA_FLAGS := -m
USE_WANDB := True
SEED := 0

## DENSITY : pass multiple densities via commandline
DENSITY := 0.2

.PHONY: help

.DEFAULT: help

## install: install all dependencies
install:
	pip install -r requirements.txt --upgrade
	pip install -e .

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