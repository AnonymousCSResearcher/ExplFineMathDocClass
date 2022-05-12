import evaluation as eval
import pandas as pd
import os
import json

# FIX

# set paths
inpath = r'C:\Users\phili\Downloads'
#filepath = r'C:\Users\Lenovo 17\Downloads'
filename_input = 'out.csv'#full.csv
fullpath = os.path.join(inpath,filename_input)
#outpath = 'evaluation/alldocs/ngrams_2-3/'

# set parameter
#nr_docs = 100
train_split_rate = 0.7

# EXECUTE

# Load table
table = pd.read_csv(fullpath,delimiter=',')
total_docs = len(table)
train_split_docs = int(total_docs*train_split_rate)
nr_docs = train_split_docs
cls_ent_idx,ent_cls_idx = eval.generate_msc_keyword_mapping(table,nr_docs)

with open('cls_ent_idx_split.json', 'w') as f:
    json.dump(cls_ent_idx, f)
with open('ent_cls_idx_split.json', 'w') as f:
    json.dump(ent_cls_idx, f)

print()