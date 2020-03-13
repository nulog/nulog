#!/usr/bin/env python
# coding: utf-8
import hashlib
import os
import sys



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pandas as pd
import re
from tqdm import trange
import random

from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset, DataLoader

# In[25]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision import transforms, utils



# !/usr/bin/env python

def p_percentage(data_path, fraction=0.1):
    with open(data_path, "r") as file:
        content = file.readlines()

    with open(data_path[:-4]+str(fraction).replace(".", "__")+data_path[-4:], "w") as file1:
        file1.writelines(content[:int(0.1*len(content))])


def load_data(log_format, path, logName):
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(os.path.join(path, logName), regex, headers, log_format)
    return df_log

def log_to_dataframe(log_file, regex, headers, logformat):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

class Anomaly_Detector_FineTuner(nn.Module):

    def __init__(self, loaded_model, new_generator):
        super(Anomaly_Detector_FineTuner, self).__init__()
        #self.hidden_size = hidden_size
        #self.target_size = target_size
        self.ff1 = nn.Linear(self.hidden_size, self.target_size)
        self.loaded_model = loaded_model


    def forward(self, log_embeddings):
        print(log_embeddings.shape())
        self.ff1(log_embeddings)


class Anomaly_Detection:
    def __init__(self, anomaly_detector):
        self.model = anomaly_detector


    def create_train_dataloader(self, descriptive_training_data, ground_truth_training_data):
        return 0

    def create_test_dataloader(self, descriptive_training_data, ground_truth_training_data):
        return 0

    def train(self, training_data, batch_size, learning_rate, optimizer, criteria, number_epochs):
        return 0

    def post_processing(self):
        return 0


sys.path.append('../')
from logparser.AttentionParser import AttentionParser
from logparser.utils import evaluator

import os
import pandas as pd

input_dir = '/home/matilda/PycharmProjects/AIOps/3_Preprocessed_Data/Logs/logs/'  # The input directory of log file
output_dir = '/home/matilda/PycharmProjects/AIOps/4_Analysis/Logs/res_logs/'  # The output directory of parsing results


log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
path = "/home/matilda/PycharmProjects/AIOps/3_Preprocessed_Data/Logs/experiments_anomaly_det/"
logName = 'BGL0__1.log'

dataFrame = load_data(log_format, path, logName)


ground_truth = np.where(dataFrame.Label == dataFrame.Label[0], 0, 1)


#ground_truth[:10] = 1 # DELETE THISs
#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#ground_truth =  ohe.fit_transform(ground_truth.reshape(-1, 1))
#ground_truth = torch.tensor(np.array(ground_truth.todense().astype(np.float   ).tolist()))

input_dir = path
output_dir = input_dir

benchmark_settings = {

    'BGL': {
        'log_file': 'BGL0__1.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'filters': '([ |:|\(|\)|=|,])|(core.)|(\.{2,})',
        'k': 50,
        'nr_epochs': 3,
        'num_samples': 0,
        'batch_size': 5
    },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'filters': '([ |:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;])',
    #     'k': 25,
    #     'nr_epochs': 5,
    #     'num_samples': 5000
    # },
    #
    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'filters': '([ |:|\(|\)|"|\{|\}|@|$|\[|\]|\||;])',
    #     'k': 5,
    #     'nr_epochs': 6,
    #     'num_samples': 0
    #
    # },
    #
    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'filters': '(\s+blk_)|(:)|(\s)',
    #     'k': 15,
    #     'nr_epochs': 5,
    #     'num_samples': 0
    # },
    #
    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'filters': '([ ])',
    #     'k': 12,
    #     'nr_epochs': 5,
    #     'num_samples': 0
    # },
    #
    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'filters': '([ |=])',
    #     'num_samples': 0,
    #     'k': 10,
    #     'nr_epochs': 3
    # },
    #
    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'filters': '([ ])',
    #     'num_samples': 0,
    #     'k': 95,
    #     'nr_epochs': 5
    # },
    #
    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'filters': '([ ])',
    #     'num_samples': 0,
    #     'k': 100,
    #     'nr_epochs': 5
    # },
    #
    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'filters': '([ ])|([\w-]+\.){2,}[\w-]+',
    #     'num_samples': 0,
    #     'k': 300,
    #     'nr_epochs': 10
    # },
    #
    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    #     'filters': '([ ])|(\d+\sB)|(\d+\sKB)|(\d+\.){3}\d+|\b[KGTM]?B\b|([\w-]+\.){2,}[\w-]+',
    #     'num_samples': 0,
    #     'k': 50,
    #     'nr_epochs': 3
    # },
}

# benchmark_settings = {

#     'BGL': {
#         'log_file': 'BGL/BGL_2k.log',
#         'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
#         'filters': '([ |:|\(|\)|=|,])|(core.)|(\.{2,})',
#         'k': 50,
#         'nr_epochs':3,
#         'num_samples':0
#     },

#     'Andriod': {
#         'log_file': 'Andriod/Andriod_2k.log',
#         'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
#         'filters': '([ |:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;])',
#         'k': 6,
#         'nr_epochs':3,
#         'num_samples':0
#     },

#     'OpenStack': {
#         'log_file': 'OpenStack/OpenStack_2k.log',
#         'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
#         'filters' : '([ |:|\(|\)|"|\{|\}|@|$|\[|\]|\||;])',
#         'k':5,
#         'nr_epochs':6,
#         'num_samples':0

#     },

#     'HDFS': {
#         'log_file': 'HDFS/HDFS_2k.log',
#         'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
#         'filters': '(\s+blk_)|(:)|(\s)',
#         'k':15,
#         'nr_epochs':10,
#         'num_samples':0
#         },

#     'Apache': {
#         'log_file': 'Apache/Apache_2k.log',
#         'log_format': '\[<Time>\] \[<Level>\] <Content>',
#         'filters': '([ ])',
#         'k':12,
#         'nr_epochs':5,
#         'num_samples':0
#         },

#     'HPC': {
#         'log_file': 'HPC/HPC_2k.log',
#         'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
#         'filters': '([ |=])',
#         'num_samples':0,
#         'k':10,
#         'nr_epochs':3
#         },

#     'Windows': {
#         'log_file': 'Windows/Windows_2k.log',
#         'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
#         'filters': '([ ])',
#         'num_samples':0,
#         'k':95,
#         'nr_epochs':5
#         },

#     'HealthApp': {
#         'log_file': 'HealthApp/HealthApp_2k.log',
#         'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
#         'filters': '([ ])',
#         'num_samples':0,
#         'k':100,
#         'nr_epochs':5
#         },

#     'Mac': {
#         'log_file': 'Mac/Mac_2k.log',
#         'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
#         'filters': '([ ])|([\w-]+\.){2,}[\w-]+',
#         'num_samples':0,
#         'k':300,
#         'nr_epochs':10
#         },

#     'Spark': {
#         'log_file': 'Spark/Spark_2k.log',
#         'log_format': '<Date> <Time> <Level> <Component>: <Content>',
#         'filters': '([ ])|(\d+\sB)|(\d+\sKB)|(\d+\.){3}\d+|\b[KGTM]?B\b|([\w-]+\.){2,}[\w-]+',
#         'num_samples':0,
#         'k':50,
#         'nr_epochs':3
#         },
# }



# for m in range(1):
#     bechmark_result = []
#     for dataset, setting in benchmark_settings.items():
#         print('\n=== Evaluation on %s ===' % dataset)
#         indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
#         log_file = os.path.basename(setting['log_file'])
#
#         parser = AttentionParser.LogParser(indir=indir, outdir=output_dir, filters=setting['filters'], k=setting['k'],
#                                            log_format=setting['log_format'])
#         parser.parse(log_file, nr_epochs=setting['nr_epochs'], num_samples=setting['num_samples'], batch_size=setting['batch_size'])
#
#         accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std = evaluator.evaluate(
#             groundtruth=os.path.join(indir, log_file + '_structured.csv'),
#             parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
#         )
#         bechmark_result.append(
#             [dataset, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std])
#
#     print('\n=== Overall evaluation results ===')
#     df_result = pd.DataFrame(bechmark_result,
#                              columns=['Dataset', 'Accuracy_PA', 'Accuracy_ExactMatching', 'Edit_distance_mean',
#                                       'Edit_distance_std'])
#     df_result.set_index('Dataset', inplace=True)
#     print(df_result)
#     df_result.T.to_csv(output_dir + 'AttentionParser_bechmark_result_run_' + str(m) + '.csv')


# data_path = "/home/matilda/PycharmProjects/AIOps/3_Preprocessed_Data/Logs/experiments_anomaly_det/BGL.log"

# p_percentage(data_path)


for dataset, setting in benchmark_settings.items():
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    parser = AttentionParser.LogParser(indir=indir, outdir=output_dir, filters=setting['filters'], k=setting['k'],
                                       log_format=setting['log_format'])

    print("STARTING FINE TUNINNING")
    parser.fine_tune(log_file, "/home/matilda/PycharmProjects/AIOps/3_Preprocessed_Data/Logs/experiments_anomaly_det/model_parser_BGL0__1.log2.pt", ground_truth, batch_size=5, nr_epochs=2)

