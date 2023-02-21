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

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)
    parser.add_argument('--apply_rule_base', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def detect_multi_spoiler(question, paragraphs):
    points = 0

    passage = ""
    for i in paragraphs:
        passage += i

    if re.match(".*\d+\s*[\.\)].+\d+?\s*[\.\)].+?\d+\s*[\.\)]", passage, re.MULTILINE | re.IGNORECASE):
        points += 3

        if re.match("^\d", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        if re.match("These are", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        if re.match("[\.\?\!\s\d\s]", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        if re.match(".*these \d", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        if re.match("need to know", question, re.MULTILINE | re.IGNORECASE):
            points += 3

        if re.match("need to know", question, re.MULTILINE | re.IGNORECASE):
            points += 3

    if points >= 6:
        return True

    else:
        return False


import re


def extract_enumeration_spoiler(paragraphs):
    passage = ""
    for i in paragraphs:
        passage += i

    enum2 = []
    for i in paragraphs:
        m = re.search("[1-9]\d{0,1}\s*[\.\)]\s.+", i, re.MULTILINE)
        if m:
            enum2.append(m.group(0))
    if len(enum2) >= 5:
        if enum2[0].startswith("1"):
            enum2 = enum2[:5]
        else:
            enum2 = enum2[-5:]
    else:
        enum2 = enum2

    # if spoiler == enum2:
    #    print("success")

    # print("\n-------------\n")
    return enum2


def predict_2(inputs, model):
    for i in inputs:
        if detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]):
            spoiler = extract_enumeration_spoiler(i["targetParagraphs"])
            if len(spoiler) > 0:
                yield {'uuid': i['uuid'], 'spoiler': spoiler}
            else:
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
                confidence = 0
                answer = ""
                for k in result[0].prediction:
                    # print(k.answer, k.confidence)
                    if confidence < k.confidence:
                        confidence = k.confidence
                        answer = k.answer

                yield {'uuid': i['uuid'], 'spoiler': answer}

        elif detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]) == False:
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
            confidence = 0
            answer = ""
            for k in result[0].prediction:
                # print(k.answer, k.confidence)
                if confidence < k.confidence:
                    confidence = k.confidence
                    answer = k.answer

            yield {'uuid': i['uuid'], 'spoiler': answer}

def predict_1(inputs, model):
    for i in inputs:
        #if detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]):
        #    spoiler = extract_enumeration_spoiler(i["targetParagraphs"])
        #    yield {'uuid': i['uuid'], 'spoiler': spoiler}

        #elif detect_multi_spoiler(i["postText"][0], i["targetParagraphs"]) == False:
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
        confidence = 0
        answer = ""
        for k in result[0].prediction:
            # print(k.answer, k.confidence)
            if confidence < k.confidence:
                confidence = k.confidence
                answer = k.answer

        yield {'uuid': i['uuid'], 'spoiler': answer}

def run_baseline(input_file, output_file, apply_rule_base , model):
    print("Generating Spoilers...")
    with open(input_file, 'r', encoding="utf-8") as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        #inp = inp[:2]
        if apply_rule_base == "v1":
            for output in predict_1(inp, model):
                print(output)
                out.write(json.dumps(output) + '\n')
        elif apply_rule_base == "v2":
            for output in predict_2(inp, model):
                print(output)
                out.write(json.dumps(output) + '\n')

if __name__ == '__main__':
    inferencer = QAInferencer.load("/saved_models/qa-model-task2", batch_size=40, gpu=True,
                                   task_type="question_answering")

    args = parse_args()

    run_baseline(args.input, args.output, args.apply_rule_base, inferencer)

    with open(args.output, 'r', encoding="utf-8") as result:
         result = [json.loads(i) for i in result]
    print(len(result))
    """import sys
    print('\n'.join(sys.path))"""

    #run_baseline("../data/validation.jsonl", "../data/output.jsonl", "v2", inferencer)