import pandas as pd
import json
from collections import Counter
from get_ConceptClassSpaces import get_de, get_mscs, get_keywords, get_refs
import math
import numpy as np

# set paths
data_path = "C:\\Users/phili/Downloads/out.csv"
dict_path = 'evaluation/alldocs/ngrams_2-3/ent_cls_idx_split.json'
eval_path = 'evaluation/classification/keywords_vs_refs_mrmscs.csv'
mrms_path = 'C:\\Users/phili/Downloads/msc-mapping-zbmath-ams.csv'

# set parameter
nr_mscs_cutoff = 10
test_split = 0.7

def load_data():
    # load data
    # msc(keyword) index
    with open(dict_path, 'r') as f:
        keyword_msc_index = json.load(f)
    # raw data
    raw_data = pd.read_csv(data_path)

    return raw_data,keyword_msc_index

def get_mrmscs_dict():

    try:
        with open('../mrmscs_dict.json', 'r') as f:
            mrmscs_dict = json.load(f)

    except:
        mrmscs_table = pd.read_csv(mrms_path, sep=';')
        mrmscs_dict = {}
        for idx,_ in mrmscs_table.iterrows():
            de = mrmscs_table['zbmath-id'][idx]
            mscs = mrmscs_table['mr-msc'][idx]
            mscs = mscs.replace('(',' ').replace(')','').split()
            mrmscs_dict[str(de)] = mscs

            #print(str(round(idx/len(mrmscs_table) * 100, 2)) + '%')

        with open('../mrmscs_dict.json', 'w') as f:
            json.dump(mrmscs_dict,f)

    return mrmscs_dict

def get_DCG(actual_mscs, predicted_mscs):

    i_max = len(actual_mscs)
    j_max = len(predicted_mscs)

    DCGs = []
    for i in range(i_max):
        msc_actual = actual_mscs[i]
        DCG = 0
        for j in range(j_max):
            msc_predicted = predicted_mscs[j]
            if msc_actual == msc_predicted:
                # score and rank
                if i == 1:
                    score = 2
                else:
                    score = 1
                rank = j+1
                # DCG
                DCG += score / math.log2(rank+1)
        DCGs.append(DCG)

    # average over actual mscs
    if len(DCGs) != 0:
        DCG = np.mean(DCGs)
    elif len(DCGs) == 0:
        DCG = 0

    return DCG

def get_dcg_table(raw_data,mrmscs_dict,keyword_msc_index):

    # predict and evaluate
    row_list = []
    tot_rows = len(raw_data)
    nr_docs_cutoff = int(tot_rows*test_split)
    na_counter = 0
    for idx,_ in raw_data.iterrows():

        if idx > nr_docs_cutoff:
            # get row content

            # get de/mscs
            de = get_de(raw_data,idx)
            mscs = get_mscs(raw_data,idx)
            try:
                mrmscs = mrmscs_dict[str(de)][:nr_mscs_cutoff]
            except:
                mrmscs = []
                na_counter += 1

            # proceed only if mscs and mrmscs available
            if len(mscs) > 0 and len(mrmscs) > 0: # True

                # get keyword mscs
                keywords = get_keywords(raw_data,idx)
                keywords_mscs = []
                for keyword in keywords:
                    try:
                        keywords_mscs.extend(keyword_msc_index[keyword])
                    except:
                        pass
                keywords_mscs = list(Counter(keywords_mscs[:nr_mscs_cutoff]))

                # get reference mscs
                refs = get_refs(raw_data,idx)
                refs_mscs = list(Counter(refs))[:nr_mscs_cutoff]

                # get intersection and union of keyword and reference mscs
                keyword_and_refs_mscs = list(set(keywords_mscs).intersection(set(refs_mscs)))
                keyword_or_refs_mscs = list(set(keywords_mscs).union(set(refs_mscs)))

                # populate evaluation table
                # get nDCGs
                # ideal DCG for normalization
                IDCG = get_DCG(mscs,mscs)
                # other nDCGs (mrmscs, keywords, refs)
                nDCG_mrmscs = get_DCG(mscs,mrmscs)/IDCG
                nDCG_keywords = get_DCG(mscs,keywords_mscs)/IDCG
                nDCG_refs = get_DCG(mscs,refs_mscs)/IDCG
                nDCG_keywords_and_refs = get_DCG(mscs,keyword_and_refs_mscs)/IDCG
                nDCG_keywords_or_refs = get_DCG(mscs,keyword_or_refs_mscs)/IDCG

                # append and save evaluation table
                new_row = {'de': de, 'mscs': mscs, 'mrmscs': mrmscs,
                                 'keyword_mscs': keywords_mscs,
                                 'refs_mscs': refs_mscs,
                            'nDCG_mrmscs': nDCG_mrmscs,
                            'nDCG_keywords': nDCG_keywords,
                            'nDCG_refs': nDCG_refs,
                           'nDCG_keywords_and_refs': nDCG_keywords_and_refs,
                           'nDCG_keywords_or_refs': nDCG_keywords_or_refs}
                row_list.append(new_row)

                # save
                #eval_table = pd.DataFrame(row_list)
                #eval_table.to_csv(eval_path)

                # print result and/or progress
                #print(new_row)
                #print(str(round(idx/tot_rows*100,2)) + '%')

    print('Matching mscs/mrmscs: ' + str((1-na_counter/tot_rows)*100) + '%')

    # save
    eval_table = pd.DataFrame(row_list)
    eval_table.to_csv(eval_path)

def compare_DCGs(eval_table):

    # mrmscs
    list_nDCG_mrmscs = list(eval_table['nDCG_mrmscs'])
    mean_nDCG_mrmscs = np.mean(list_nDCG_mrmscs)
    print('mean_nDCG_mrmscs: ' + str(mean_nDCG_mrmscs))

    # keywords
    list_nDCG_keywords = list(eval_table['nDCG_keywords'])
    mean_nDCG_keywords = np.mean(list_nDCG_keywords)
    print('mean_nDCG_keywords: ' + str(mean_nDCG_keywords))

    # refs
    list_nDCG_refs = list(eval_table['nDCG_refs'])
    mean_nDCG_refs = np.mean(list_nDCG_refs)
    print('mean_nDCG_refs: ' + str(mean_nDCG_refs))

    # keywords AND refs
    list_nDCG_keywords_and_refs = list(eval_table['nDCG_keywords_and_refs'])
    mean_nDCG_keywords_and_refs = np.mean(list_nDCG_keywords_and_refs)
    print('mean_nDCG_keywords_and_refs: ' + str(mean_nDCG_keywords_and_refs))

    # keywords OR refs
    list_nDCG_keywords_or_refs = list(eval_table['nDCG_keywords_or_refs'])
    mean_nDCG_keywords_or_refs = np.mean(list_nDCG_keywords_or_refs)
    print('mean_nDCG_keywords_or_refs: ' + str(mean_nDCG_keywords_or_refs))

# get eval table
print('Load data')
raw_data,keyword_msc_index = load_data()
mrmscs_dict = get_mrmscs_dict()
print('Get DCGs')
get_dcg_table(raw_data,mrmscs_dict,keyword_msc_index)

# eval eval table
eval_table = pd.read_csv(eval_path)
compare_DCGs(eval_table)

print('end')