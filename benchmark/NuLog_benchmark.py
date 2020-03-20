#!/usr/bin/env python

import sys

sys.path.append('../')
from logparser.NuLog import NuLogParser
from logparser.utils import evaluator

import os
import pandas as pd

input_dir = '../logs/'  # The input directory of log file
output_dir = './AttentionParserResult/'  # The output directory of parsing results

benchmark_settings = {

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'filters': '([ |:|\(|\)|=|,])|(core.)|(\.{2,})',
        'k': 50,
        'nr_epochs':3,
        'num_samples':0
    },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'filters': '([ |:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;])',
        'k': 25,
        'nr_epochs':5,
        'num_samples':5000
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'filters' : '([ |:|\(|\)|"|\{|\}|@|$|\[|\]|\||;])',
        'k':5,
        'nr_epochs':6,
        'num_samples':0
            
    },
    
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'filters': '(\s+blk_)|(:)|(\s)',
        'k':15,
        'nr_epochs':5,
        'num_samples':0
        },
    
    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'filters': '([ ])',
        'k':12,
        'nr_epochs':5,
        'num_samples':0
        },
    
    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'filters': '([ |=])',
        'num_samples':0,
        'k':10,
        'nr_epochs':3
        },
    
    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'filters': '([ ])',
        'num_samples':0,
        'k':95,
        'nr_epochs':5   
        },
    
    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'filters': '([ ])',
        'num_samples':0,
        'k':100,
        'nr_epochs':5 
        },
    
    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'filters': '([ ])|([\w-]+\.){2,}[\w-]+',
        'num_samples':0,
        'k':300,
        'nr_epochs':10  
        },
    
    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'filters': '([ ])|(\d+\sB)|(\d+\sKB)|(\d+\.){3}\d+|\b[KGTM]?B\b|([\w-]+\.){2,}[\w-]+',
        'num_samples':0,
        'k':50,
        'nr_epochs':3
        },
}


for m in range(10):
    bechmark_result = []
    for dataset, setting in benchmark_settings.items():
        print('\n=== Evaluation on %s ===' % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])

        parser = NuLogParser.LogParser(indir=indir, outdir=output_dir, filters=setting['filters'], k=setting['k'], log_format=setting['log_format'])
        parser.parse(log_file, nr_epochs=setting['nr_epochs'], num_samples=setting['num_samples'])

        accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std = evaluator.evaluate(
            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        bechmark_result.append([dataset, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std])

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'Accuracy_PA', 'Accuracy_ExactMatching','Edit_distance_mean', 'Edit_distance_std'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv(output_dir+'NuLog_benchmark_result_run_'+str(m)+'.csv')
