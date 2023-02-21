#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
import re
from simpletransformers.classification import ClassificationModel
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    parser.add_argument('--apply_ner', type=str, help='Calculate NER entities on input.', required=False)


    return parser.parse_args()


def extract_ner(df):
    import spacy
    print("Starting NER Extraction...")
    # NER = spacy.load("en_core_web_lg")
    NER = spacy.load("en_core_web_md")

    orgs = []
    persons = []
    dates = []
    locations = []

    for i in df["full_context"]:
        ner_object = NER(i)
        orgs_temp = []
        persons_temp = []
        dates_temp = []
        locations_temp = []
        for j in ner_object.ents:
            if j.label_ == "ORG":
                orgs_temp.append(str(j))
            if j.label_ == "PERSON":
                persons_temp.append(str(j))
            if j.label_ == "GPE":
                locations_temp.append(str(j))
            if j.label_ == "DATE":
                dates_temp.append(str(j))
        orgs.append(list(set(orgs_temp)))
        persons.append(list(set(persons_temp)))
        dates.append(list(set(dates_temp)))
        locations.append(list(set(locations_temp)))

    df["ner_orgs"] = orgs
    df["ner_persons"] = persons
    df["ner_dates"] = dates
    df["ner_locations"] = locations
    print("Finished NER Extraction...")
    return df


def load_input(df, apply_ner):
    print(df)
    print(type(df))
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)

    #df  = df[:50]

    # print(len(df))
    df["title_spoiler_ratio"] = df.apply(
        lambda x: len(str(x["targetParagraphs"]).split(" ")) / len(str(x["postText"]).split(" ")), axis=1)
    #df["first_spoiler"] = df.apply(lambda x: x["spoilerPositions"][0][0][0], axis=1)

    string_list = []
    for i, j in df.iterrows():
        string = ""
        for k in j["targetParagraphs"]:
            string += k + " "
        string_list.append(string)
    df["full_context"] = string_list

    if apply_ner == "yes":
        df = extract_ner(df)

    ret = []

    for _, i in df.iterrows():

        combined = ''
        combined += f'Title: {i["postText"][0]}. ' \
                    f'Spoiler Length Ratio: {i["title_spoiler_ratio"]}. '
                    #f'Spoiler: {str(i["targetParagraphs"])[1:-1]}. ' \
                    #f'First Spoiler: {i["first_spoiler"]}. ' \

        if re.match(".*\d+\s*[\.\)].+\d+?\s*[\.\)].+?\d+\s*[\.\)]", i["full_context"], re.MULTILINE | re.IGNORECASE):
            combined += f'Enumeration or multi-line. '

        """if len(i["ner_orgs"]) > 0:
            combined += f'{len(i["ner_orgs"])} organisations. '
            # combined += f'The context contains the following organisations {str(row["ner_orgs"])[1:-1]}. '
        if len(i["ner_persons"]) > 0:
            combined += f'{len(i["ner_persons"])} persons. '
            # combined += f'The context contains the following persons {str(row["ner_persons"])[1:-1]}. '
        if len(i["ner_dates"]) > 0:
            combined += f'{len(i["ner_dates"])} dates. '
            # combined += f'The context contains the following dates {str(row["ner_dates"])[1:-1]}. '
        if len(i["ner_locations"]) > 0:
            combined += f'{len(i["ner_locations"])} locations. '"""
            # combined += f'The context contains the following locations {str(row["ner_locations"])[1:-1]}. '

        
        combined += f'Publishing Platform: {i["postPlatform"]}. ' \
                    f'Source Website {i["targetUrl"]}. '
        combined = combined.replace('"', "'")

        ret += [{'text': combined, 'uuid': i['uuid']}]
    print("Finished loading DF....")
    # print(len(ret))
    return pd.DataFrame(ret)


def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def predict(df, apply_ner):
    

    labels = ['phrase', 'passage', 'multi']
    print("Load Model...")
    model = ClassificationModel('roberta', '/saved_models/roberta_onnx', use_cuda=use_cuda())
    print("Finished Loading Model...")
    df = load_input(df, apply_ner)

    uuids = list(df['uuid'])
    texts = list(df['text'])
    print("Start predictions...")
    predictions = model.predict(texts)[0]
    print(predictions)
    for i in range(len(df)):
        if predictions[i] == 0:
            yield {'uuid': uuids[i], 'spoilerType': "multi"}
        if predictions[i] == 1:
            yield {'uuid': uuids[i], 'spoilerType': "passage"}
        if predictions[i] == 2:
            yield {'uuid': uuids[i], 'spoilerType': "phrase"}


def run_baseline(input_file, output_file, apply_ner):
    print("Starting prediction...")
    with open(output_file, 'w') as out:
        for prediction in predict(input_file, apply_ner):
            print(prediction)
            out.write(json.dumps(prediction) + '\n')
    print("Finished predictions...")
            
    # with open(input_file, 'r', encoding="utf-8") as inp, open(output_file, 'w') as out:
    #    inp = [json.loads(i) for i in inp]
    #    inp = inp[:10]
    #    for output in predict(inp):
    #        out.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    print("Starting File!")
    args = parse_args()
    
    run_baseline(args.input, args.output, args.apply_ner)
    # run_baseline("../data/train.jsonl", "../data/out.jsonl")