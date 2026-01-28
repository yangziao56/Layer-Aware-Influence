import numpy as np
import pandas as pd


tasks = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'qnli', 'rte', 'wnli', 'mnli_0', 'mnli_1']
task_label = {'cola':2, 'sst2':2, 'mrpc':2, 'stsb':1, 'qqp':2, 'mnli':3, 'qnli':2, 'rte':2, 'wnli':2, 'ax':3}
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ax": ("premise", "hypothesis"),
}

task_to_file_name = {
    "cola": 'CoLA',
    "mnli": 'MNLI',
    "mrpc": 'MRPC',
    "qnli": 'QNLI',
    "qqp": 'QQP',
    "rte": 'RTE',
    "sst2": 'SST-2',
    "stsb": 'STS-B',
    "wnli": 'WNLI',
    "mnli_0": 'MNLI-mm',
    "mnli_1": 'MNLI-m',
    "ax": 'AX'
}



for task_name in tasks:
    cur_pred = pd.read_csv('results/GLUE_test/' + task_to_file_name[task_name] + '.tsv', sep='\t', index_col=None)

    cur_label = cur_pred['label']
    print(task_name, np.min(cur_label), np.max(cur_label))
