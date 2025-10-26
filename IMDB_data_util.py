import numpy as np
import pandas as pd
import icecream
import math
import sklearn
import os
import time
import gc
import random
import re
import csv
# import spacy
import torch
import torchtext
# from torchtext.datasets import IMDB
from torch import Tensor
import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.utils.data import dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from itertools import *
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram
# from tempfile import TemporaryDirectory
from typing import Tuple
from captum.concept import TCAV
# from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
# import torchdata.datapipes as dp
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr._core.layer.layer_activation import LayerActivation
# from suicidedataset import SuicideDataSet, bert_tokenizer
# from sklearn.cluster import MiniBatchKMeans
# from PIL import Image
# from wordcloud import WordCloud
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from sucideClassifier import TextClassificationModel
import pickle
from tqdm import tqdm

import imdb_bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, max_length=512)
def bert_tokenizer(text: str) -> Tuple[Tensor, Tensor]:
    tokens = tokenizer.__call__(text, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    return tokens['input_ids'], tokens['attention_mask'], words
def encode_labels(labels) -> Tensor:
    return F.one_hot(torch.tensor([y for y in labels])).to(torch.float32)


from datasets import load_dataset
def load_imdb():
    print("Loading IMDB dataset...")

    try:
        # Load IMDB dataset
        dataset = load_dataset('imdb')

        # Extract texts and labels
        X_train = np.array(dataset['train']['text'])
        y_train = np.array(dataset['train']['label'])
        X_test = np.array(dataset['test']['text'])
        y_test = np.array(dataset['test']['label'])

        # Use a subset for faster training (remove these lines to use full dataset)
        # train_texts = train_texts[:1000]  # Reduced for debugging
        # train_labels = train_labels[:1000]
        # test_texts = test_texts[:2000]
        # test_labels = test_labels[:2000]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Debug: Check label distribution
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        return  X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # return




#
#
# X_test_neg = X_test[np.invert(y_test.astype(bool))]
# X_test_pos = X_test[y_test.astype(bool)]
#
#
#
# start_index=0
# end_index = X_test_neg.shape[0] if X_test_neg.shape[0] < X_test_pos.shape[0] else X_test_pos.shape[0]
def prepare_no_labels(data,start_idx=None, end_idx=None,device='cpu'):
    input_ids = []
    attention_masks = []
    for t in data[start_idx:end_idx]:
        tokens = bert_tokenizer(t)
        input_ids.append(tokens[0])
        attention_masks.append(tokens[1])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # print(input_ids.shape, attention_masks.shape)
    return (input_ids.to(device), attention_masks.to(device))
#
# neg_ds =prepare_no_labels(X_test_neg, start_index, end_index)
# pos_ds =prepare_no_labels(X_test_pos, start_index, end_index)
#
#
# pos_dataset = torch.utils.data.TensorDataset(pos_ds[0].to(device), pos_ds[1].to(device))
# neg_dataset = torch.utils.data.TensorDataset(neg_ds[0].to(device), neg_ds[1].to(device))
#
#
#
#
# test_dataloader_pos = DataLoader(
#             pos_dataset,
#             sampler = None,
#             batch_size = 64,
#             shuffle=False
#             # num_workers= 4
#         )
# test_dataloader_neg = DataLoader(
#             neg_dataset,
#             sampler = None,
#             batch_size = 64,
#             shuffle=False
#             # num_workers= 4
#         )