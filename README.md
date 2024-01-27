# Multiple Choice Question Answering Artifacts (IN PROGRESS)

This repository is the official implementation of the in-progress paper: "Large Language Models Exploit Artifacts in Multiple Choice Question Answering"

<p align="center">
  <img src="/images/figure.png"></img>
</p>

## Overview

This repository contains the code and dataset to run the direct answer and process of elimination strategies, with and without chain of thought, on our four tested commonsense reasoning and scientific reasoning multiple-choice QA datasets.

## Setup

Python 3.10.0, pip 23.2.1, and conda 23.5.0 were used when running the code in this repository. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```

The most important files in this repository are as follows:
* `/model/`: Contains the code for running all experiments with LLMs
* `/prompts/`: Contains the prompts used in the independent full and indepdent choices-only prompt experiments
* `/evaluation/`: Contains the code for generating our plots
* `/scripts/`: Sample bash scripts to run our code

## Model Usage
Nugget
There are four relevant files:
* `/model/run_hf.py`: Code to run all possible prompt formats, except the second step of the 
* `/model/run_hf_question.py`: Contains the code for generating our plots
* `/scripts/`: Sample bash scripts to run our code

## Evaluation Usage
