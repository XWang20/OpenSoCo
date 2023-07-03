import os
from utils import *
import pandas as pd

def get_dataset(file_path, do_preprocess):
    labels = []
    contents = []
    with open(file_path, 'r')as f:
        raw = f.readlines()
        for line in raw:
            # contents.append(normalizeTweet(line.strip().split("\t")[0]))
            labels.append(line.strip().split("\t")[-1])
    label_list=list(set(labels))
    label_to_id={label:i for i,label in enumerate(label_list)}
    labels = []
    with open(file_path, 'r')as f:
        raw = f.readlines()
        for line in raw:
            if do_preprocess == "none":
                text = normalizeTweet("".join(line.strip().split("\t")[:-1])) 
            else:
                text = preprocess_clean("".join(line.strip().split("\t")[:-1])) 
            contents.append(text)
            labels.append(label_to_id[line.strip().split("\t")[-1]])
    assert(len(contents)==len(labels))
    return contents, labels

def get_multi_label_dataset(file_path, do_preprocess, train_sample_num=1):
    labels = []
    contents = []
    df = pd.read_csv(file_path,lineterminator='\n')
    if train_sample_num != 1 and train_sample_num<=len(df):
        df = df.sample(n=train_sample_num)
    for _, row in df.iterrows():
        if do_preprocess == "none":
            contents.append(normalizeTweet(row["text"]))
        else:
            contents.append(preprocess_clean(row["text"]))
        labels.append(list(row[2:].values))
    assert(len(contents)==len(labels))
    return contents, labels

def get_single_label_dataset(file_path, do_preprocess, seed, train_sample_num=1):
    labels = []
    contents = []
    df = pd.read_csv(file_path,lineterminator='\n')
    if "text_a" in df.keys() and "text_b" in df.keys():
        contents, labels = get_two_sentence_dataset(file_path, do_preprocess, train_sample_num)
        return contents, labels
    if train_sample_num != 1 and train_sample_num<=len(df):
        df = df.sample(n=train_sample_num)
    if "target" in df.keys():
        df = df.drop(["target"],axis=1)

    for _, row in df.iterrows():
        if do_preprocess == "bertweet":
            contents.append(normalizeTweet(str(row["text"])))
        else:
            contents.append(preprocess_clean(str(row["text"])))
        for i, value in enumerate(list(row[2:].values)):
            if value==1:
                labels.append(i)
                break
    assert(len(contents)==len(labels))
    return contents, labels

def get_NEP_label_dataset(file_path, do_preprocess, train_sample_num=1):
    labels = []
    contents = []
    df = pd.read_csv(file_path)
    if train_sample_num != 1 and train_sample_num<=len(df):
        df = df.sample(n=train_sample_num)
    for _, row in df.iterrows():
        if do_preprocess == "none":
            contents.append(normalizeTweet(str(row["text"])))
        else:
            contents.append(preprocess_clean(str(row["text"])))
        for i, value in enumerate(list(row[2:].values)):
            if value==1:
                labels.append(1-i)
                break
    # import pdb;pdb.set_trace()
    assert(len(contents)==len(labels))
    return contents, labels

def get_stance_dataset(file_path, do_preprocess, train_sample_num=1):
    labels = []
    contents = []
    df = pd.read_csv(file_path)
    if train_sample_num != 1 and train_sample_num<=len(df):
        df = df.sample(n=train_sample_num)
    for _, row in df.iterrows():
        contents.append(tuple((preprocess_clean(str(row["text"])), row["target"])))
        if not row["target"]:
            print(row)
        for i, value in enumerate(list(row[3:].values)):
            if value==1:
                labels.append(i)
                break
    assert(len(contents)==len(labels))
    return contents, labels

def get_emergent_dataset(file_path, do_preprocess):
    labels = []
    contents = []
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        if do_preprocess == "none":
            contents.append(tuple((normalizeTweet(row["claim"]), normalizeTweet(row["article"]))))
        else:
            contents.append(tuple((preprocess_clean(row["claim"]), preprocess_clean(row["article"]))))
        for i, value in enumerate(list(row[3:].values)):
            if value==1:
                labels.append(i)
                break
    assert(len(contents)==len(labels))
    return contents, labels

def get_two_sentence_dataset(file_path, do_preprocess, train_sample_num=1):
    labels = []
    contents = []
    df = pd.read_csv(file_path,lineterminator='\n')
    df = df.fillna("")
    if train_sample_num != 1:
        df = df.sample(n=train_sample_num)

    for _, row in df.iterrows():
        if do_preprocess == "none":
            contents.append(tuple((normalizeTweet(row["text_a"]), normalizeTweet(row["text_b"]))))
        else:
            contents.append(tuple((preprocess_clean(row["text_a"]), preprocess_clean(row["text_b"]))))
        for i, value in enumerate(list(row[3:].values)):
            if value==1:
                labels.append(i)
                break
    assert(len(contents)==len(labels))
    return contents, labels