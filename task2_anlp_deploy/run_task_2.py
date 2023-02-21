#!/usr/bin/env python3
import argparse
import json
import re
import torch
import os
import pprint
from pathlib import Path

from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.utils import write_squad_predictions
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.language_model import LanguageModel
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.modeling.adaptive_model import AdaptiveModel
from farm.infer import QAInferencer
from farm.eval import Evaluator
from farm.evaluation.metrics import metrics_per_bin


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    # Adding commandline input parameters for the input and output path and for the model to use
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)
    parser.add_argument('--apply_rule_base', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def detect_multi_spoiler(question, paragraphs):

    # Create scoring variable, to decide if spoiler is a multi spoiler
    points = 0

    # Recreate the full context from the single paragraphs
    passage = ""
    for i in paragraphs:
        passage += i

    # Search for patterns in the question and the context to identify multi spoilers

    # Look out for enumerations in the context ( like 1. ... 2. ... )
    if re.match(".*\d+\s*[\.\)].+\d+?\s*[\.\)].+?\d+\s*[\.\)]", passage, re.MULTILINE | re.IGNORECASE):
        # If pattern matches apply +3 to the score and excute additional constrains
        points += 3

        # Condition if question starts with a number, highly likely to be a multi spoiler
        if re.match("^\d", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        # Condition if question starts with "These are"..., highly likely to be a multi spoiler
        if re.match("These are", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        # Condition if question contains a punctuation mark followed by a number
        if re.match("[\.\?\!\s\d\s]", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        # Condition if question contains a "these" followed by a number
        if re.match(".*these \d", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        # Condition if question contains a "need to know"
        if re.match("need to know", question, re.MULTILINE | re.IGNORECASE):
            points += 3

    # If score is equal or higher than six, question is considered and treated as multi spoiler
    if points >= 6:
        return True

    else:
        return False


import re


def extract_enumeration_spoiler(paragraphs):

    # Initializing a list for the collected enumerations in the context
    enum = []

    # Iterate through the single paragraphs and extract enumerations if found ( like 1. ... or 1) ... )
    for i in paragraphs:
        m = re.search("[1-9]\d{0,1}\s*[\.\)]\s.+", i, re.MULTILINE)
        # Add found enumerations to list
        if m:
            enum.append(m.group(0))

    # Condition to just cover the first 5 enumerations of a context, because the dataset always just cover the top 5
    if len(enum) >= 5:
        # Condition to ensure that Top 5 is considered, even when the listing in the context is reversed (descending)
        if enum[0].startswith("1"):
            enum = enum[:5]
        else:
            enum = enum[-5:]
    else:
        # Condition to ensure that Top 5 is considered, even when the listing in the context is reversed (descending)

        if enum[0].startswith("1"):
            enum = enum
        else:
            enum = enum.reverse()

    return enum

# Function predict_2 is an adaptive approach featuring the transformer model and a rule based regex approach
def predict_2(inputs, model):

    # Iterate through the test dataset columns
    for i in inputs:
        # Check if spoiler is a multi part spoiler
        if detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]):
            # Extract the enumerations, if spoiler is detected as multi part spoiler
            spoiler = extract_enumeration_spoiler(i["targetParagraphs"])
            # If the result list of enumerations is not empty, return the enumerations a suggested spoilers
            if len(spoiler) > 0:
                yield {'uuid': i['uuid'], 'spoiler': spoiler}
            # If the result list of enumerations is empty, apply the transformer model and return the suggested answer of the model
            else:
                # Recreate the full context from the single paragraphs
                text = ""
                for j in i["targetParagraphs"]:
                    text += j + " "
                    QA_input = [
                    {
                        "questions": i["postText"],
                        "text": text
                    }]
                # print(i["postText"][0])
                # print(text)
                result = model.inference_from_dicts(dicts=QA_input, return_json=False)

                # Extract the answer with the highest confidence score
                confidence = 0
                answer = ""

                for k in result[0].prediction:
                    # print(k.answer, k.confidence)
                    if confidence < k.confidence:
                        confidence = k.confidence
                        answer = k.answer

                yield {'uuid': i['uuid'], 'spoiler': answer}

        # If spoiler is not considered a multi part spoiler, apply transformer model approach
        elif detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]) == False:
            # Recreate the full context from the single paragraphs
            text = ""
            for j in i["targetParagraphs"]:
                text += j + " "
            QA_input = [
                {
                    "questions": i["postText"],
                    "text": text
                }]

            result = model.inference_from_dicts(dicts=QA_input, return_json=False)

            # Extract the answer with the highest confidence score

            confidence = 0
            answer = ""
            for k in result[0].prediction:
                if confidence < k.confidence:
                    confidence = k.confidence
                    answer = k.answer

            yield {'uuid': i['uuid'], 'spoiler': answer}

# Function predict_1 only consideres the transformer model in the approach
def predict_1(inputs, model):

    # Iterate through the test dataset columns

    for i in inputs:
        # Recreate the full context from the single paragraphs
        text = ""
        for j in i["targetParagraphs"]:
            text += j + " "
        QA_input = [
            {
                "questions": i["postText"],
                "text": text
            }]

        result = model.inference_from_dicts(dicts=QA_input, return_json=False)

        # Extract the answer with the highest confidence score

        confidence = 0
        answer = ""
        for k in result[0].prediction:
            if confidence < k.confidence:
                confidence = k.confidence
                answer = k.answer

        yield {'uuid': i['uuid'], 'spoiler': answer}

def run_approach(input_file, output_file, apply_rule_base , model):
    print("Generating Spoilers...")
    # Open the test dataset (read mode) and the output file (write mode) with the suggested spoilers
    with open(input_file, 'r', encoding="utf-8") as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        # Condition if approach 1 or 2 should be executed
        if apply_rule_base == "v1":
            for output in predict_1(inp, model):
                out.write(json.dumps(output) + '\n')
        elif apply_rule_base == "v2":
            for output in predict_2(inp, model):
                out.write(json.dumps(output) + '\n')

if __name__ == '__main__':
    # Load pretrained transformer model for extractive question answering
    inferencer = QAInferencer.load("/saved_models/qa-model-task2", batch_size=40, gpu=True,
                                   task_type="question_answering")
    # Access the commandline parameters
    args = parse_args()

    # Execute one of the two approaches, depending on args.apply_rule_base parameter
    run_approach(args.input, args.output, args.apply_rule_base, inferencer)

