SHELL:= /../../bin/bash
all:
	source ~/.bashrc_profile
	module load cuda-toolkit/8.0
	python run_model.py

init:
	source ~/.bashrc_profile
load:
	module load cuda-toolkit/8.0
train:
	python run_model.py

eval:
	python eval.py
