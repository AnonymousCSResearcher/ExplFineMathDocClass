import pandas as pd
import os
import json
#import pywikibot
#import SPARQLWrapper
import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk import WordNetLemmatizer
from nltk import ngrams

from collections import Counter
import math
import numpy as np
import scipy

##########
# DEFINE #
##########

# nlp

def get_stopwords():
    with open("stopwords.txt",'r') as f:
        sstopwords = f.readlines()
    return sstopwords

def remove_stopwords_custom(text):
    # remove stopwords and punctuation
    text = text.lower()
    sstopwords = get_stopwords()
    punctuations = [',',';','.','!','?',':','(',')','\\','+','-']
    for stopword in sstopwords:
        text = text.replace(stopword,"")
    for punctuation in punctuations:
        text = text.replace(punctuation,"")
    return text

def remove_stopwords_nltk(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw

def stemming_lemmatization(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def nlp_clean(text):
    text = remove_stopwords_custom(text)
    tokens_without_sw = remove_stopwords_nltk(text)
    clean_words = []
    for word in tokens_without_sw:
        clean_words.append(stemming_lemmatization(word))
    return ' '.join(clean_words)
    #return clean_words

#cleaned_text = nlp_clean("the vortices")
#print(cleaned_text)

# get Wikidata qid from name using pywikibot
def get_qid_pywikibot(name):
    try:
        site = pywikibot.Site("en", "wikipedia")
        page = pywikibot.Page(site, name)
        item = pywikibot.ItemPage.fromPage(page)
        qid = item.id
    except:
        qid = 'N/A'
    return qid

def get_sparql_results(sparql_query_string):
    sparql = SPARQLWrapper.SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query_string)
    try:
        # stream with the results in XML, see <http://www.w3.org/TR/rdf-sparql-XMLres/>
        sparql.setReturnFormat(SPARQLWrapper.JSON)
        result = sparql.query().convert()
    except:
        result = None
    return result

def get_qid_sparql(name):

    sparql_query_string = """SELECT distinct ?item ?itemLabel ?itemDescription WHERE{  
        ?item ?label "%s"@en. 
        ?article schema:about ?item .
        ?article schema:inLanguage "en" .
        ?article schema:isPartOf <https://en.wikipedia.org/>. 
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }    
        }""" % name

    sparql_results = get_sparql_results(sparql_query_string)

    qid_results = []
    try:
        for result in sparql_results['results']['bindings']:
            try:
                desc = result['itemDescription']['value']
                if desc != 'Wikimedia disambiguation page':
                    url = result['item']['value']
                    qid = url.split("/")[-1]
                    qid_results.append(qid)
                else:
                    pass
            except:
                pass
    except:
        pass

    if len(qid_results) > 0:
        qid = qid_results[0]  # take first result
    else:
        qid = 'N/A'

    return qid

def get_anchor(qid,name):
    linked = qid + ">" + name
    anchor = """<a href="https://www.wikidata.org/wiki/%s</a>""" % linked
    return anchor

def get_entity_linking_wikidata_1gram(text):

    # start timer
    start = time.time()

    link_text = ""

    # get entity linking qids
    links = []
    for word in text.split():
        print(word)
        #qid = get_qid_pywikibot(word)
        qid = get_qid_sparql(word)
        links.append((word,qid))
        print(qid)
        if qid is not None:
            link_text += get_anchor(qid,word) + " "
        else:
            link_text += word + " "

    #print(link_text)

    # stop timer
    end = time.time()
    print("Runtime: " + str(end-start))

    return link_text

def get_entity_linking_wikidata_ngram(text,n_gram_length):

    link_text = ""

    # get entity / qid candidates
    entities = []
    qids = []
    nngrams = ngrams(text.split(), n=n_gram_length)
    for nngram in nngrams:
        name = ''
        for word in nngram:
            name += word + " "
            entities.append(name[:-1])
            qid = get_qid_pywikibot(word)
            qids.append(qid)
            print(name + ": " + str(qid))
            if qid is not None:
                link_text += get_anchor(qid, word) + " "
            else:
                link_text += word + " "

    return entities,qids,link_text

file = "enwiki-latest-all-titles-in-ns0"

def get_Wikipedia_article_names(n_gram_length):

    file = "enwiki-latest-all-titles-in-ns0"

    with open(file,"r",encoding="utf-8") as f:
        lines = f.readlines()

    #names = []
    names = {}
    for line in lines:
        if len(line.split("_")) == n_gram_length:
            name = line.strip("\n").replace("_"," ").replace('"','').lower()
            # LIST
            #names.append(name)
            # DICT
            names[name] = name

    return names

def get_entity_linking_wikipedia(text,n_gram_length):
    # load Wikipedia article name candidates
    Wikipedia_article_names = get_Wikipedia_article_names(n_gram_length)

    nngrams = ngrams(text.split(),n=n_gram_length)

    # get entity candidates
    entities = []
    for nngram in nngrams:
        name = ''
        for word in nngram:
            name += word + " "
        print(name)
        try:
            entities.append(Wikipedia_article_names[name[:-1]])
        except:
            pass

    return entities

def get_entity_linking_wikipedia_wikidata(text,n_gram_length):

    # wikipedia
    print("-------------------")
    print("Wikipedia articles:")
    print("-------------------")
    entities = get_entity_linking_wikipedia(text=text, n_gram_length=n_gram_length)

    # wikidata
    print("---------------")
    print("Wikidata items:")
    print("---------------")
    for entity in set(entities):
        qid = get_qid_sparql(entity)
        if qid is not None:
            print(entity + ": " + str(qid))

def get_text_entity_linking(text,n_gram_length):

    # init csv lines and add header
    csv_lines = []
    header_line = ""
    for col_desc in ["N_gram", "relevant", "Wikipedia_URL_correct", "Wikidata_URL_correct",
                   "Wikipedia_article_Wikidump", "eval1", "Wikipedia_URL_Wikidump", "eval2",
                   "Wikidata_item_Pywikibot", "eval3", "Wikidata_URL_Pywikibot", "eval4",
                   "Wikidata_item_SPARQL", "eval5", "Wikidata_URL_SPARQL", "eval6",
                   "comment"]:
        header_line += col_desc + ";"
    header_line = header_line[:-1] + "\n"
    csv_lines.append(header_line)
    print(header_line)

    # set prefixes and default for columns
    default = "-"
    tick = "x"
    #
    relevant = default
    eval = tick
    Wikipedia_URL_correct = default
    Wikidata_URL_correct = default
    comment = default
    Wikipedia_URL_prefix = "https://en.wikipedia.org/wiki/"
    Wikidata_URL_prefix = "https://www.wikidata.org/wiki/"

    # load Wikipedia article name candidates
    Wikipedia_article_names = get_Wikipedia_article_names(n_gram_length)

    # get n-grams
    nngrams = ngrams(text.split(), n=n_gram_length)

    # get entity candidates
    for nngram in nngrams:

        # N_gram
        N_gram = ''
        for word in nngram:
            N_gram += word + " "
        N_gram = N_gram[:-1]

        N_gram_cleaned = nlp_clean(N_gram)

        # Wikipedia
        try:
            Wikipedia_article = Wikipedia_article_names[N_gram_cleaned]
            Wikipedia_URL = Wikipedia_URL_prefix + Wikipedia_article.replace(" ","_")
        except:
            Wikipedia_article = default
            Wikipedia_URL = default

        # Wikidata_Pywikibot
        Wikidata_Pywikibot_QID = get_qid_pywikibot(N_gram_cleaned)
        if Wikidata_Pywikibot_QID is not None:
            Wikidata_Pywikibot = N_gram_cleaned
            Wikidata_Pywikibot_URL = Wikidata_URL_prefix + Wikidata_Pywikibot_QID
        else:
            Wikidata_Pywikibot = default
            Wikidata_Pywikibot_URL = default

        # Wikidata_SPARQL
        Wikidata_SPARQL_QID = get_qid_sparql(N_gram_cleaned)
        if Wikidata_SPARQL_QID is not None:
            Wikidata_SPARQL = N_gram_cleaned
            Wikidata_SPARQL_URL = Wikidata_URL_prefix + Wikidata_SPARQL_QID
        else:
            Wikidata_SPARQL = default
            Wikidata_SPARQL_URL = default

        # add csv line
        csv_line = ""
        for col_cont in [N_gram,relevant,Wikipedia_URL_correct,Wikidata_URL_correct,
                       Wikipedia_article,eval,Wikipedia_URL,eval,
                       Wikidata_Pywikibot,eval,Wikidata_Pywikibot_URL,eval,
                       Wikidata_SPARQL,eval,Wikidata_SPARQL_URL,eval,
                       comment]:
            csv_line += col_cont + ";"
        csv_line = csv_line[:-1] + "\n"
        csv_lines.append(csv_line)
        print(csv_line)

    return csv_lines

# Add entity to class entity space index
def add_cls_ent(idx,cls,ent):
    try:
        idx[cls][ent] += 1
    except:
        try:
            idx[cls][ent] = 1
        except:
            idx[cls] = {}
            idx[cls][ent] = 1
    return idx

def clean(string):
    return string.replace('[','').replace(']','').replace('\\','').replace("'",'')

def get_mscs(table,idx):
    mscs = []
    for msc in clean(table['msc'][idx]).split():
        msc = msc.strip(',')
        mscs.append(msc)
    return mscs

def get_keywords(table,idx):
    keywords = []
    for keyword in clean(str(table['keyword'][idx])).split(","):
        keyword = keyword.lstrip().rstrip()
        for clea_str in [',',"'",'"',"`",'\\']:
            keyword = keyword.strip(clea_str)
        keywords.append(keyword)
    return keywords

def get_refs(table,idx):
    return clean(table['refs'][idx]).replace(',','').split()

def get_de(table,idx):
    return table['de'][idx]

def generate_msc_keyword_mapping(table,nr_docs):
    # Iterate documents to get class-entity linking index
    max_rows = nr_docs
    # Create concept class/entity space index
    cls_ent_idx = {}
    ent_cls_idx = {}
    for idx in range(max_rows):
    #for idx,row in table.iterrows():
    #for idx in range(max_rows):

        #print(idx)
        print(round(idx/max_rows*100,1),'%')

        #de = table['de'][idx]
        mscs = get_mscs(table,idx)
        keywords = get_keywords(table,idx)

        #title = table['title'][idx]
        #abstract = table['text'][idx]
        #refs = get_refs(table,idx)

        # Set entity source
        #text = title + abstract

        # 1-GRAMS
        # TODO: count entities only once?
        #entity_names_1grams = nlp_clean(text)#list(set(nlp_clean(text)))

        # 2-GRAMS
        #entity_names_2grams = ngrams(text.split(), n=2)

        # Wikipedia article entities
        #entitiy_names_wikipedia = get_entity_linking_wikipedia(text,n_gram_length=3)

        # Wikidata item entities
        #entities_wikidata = get_entity_linking_wikidata_ngram(text,n_gram_length=3)
        #entity_names_wikidata = entities_wikidata[0]
        #qids_wikidata = entities_wikidata[1]

        # Select entity names (set source)
        entity_names = keywords#entity_names_2grams

        # Add to index
        for ent in entity_names:
            for msc in mscs:
                cls_ent_idx = add_cls_ent(cls_ent_idx,msc,ent)
                ent_cls_idx = add_cls_ent(ent_cls_idx,ent,msc)

        pass
    return cls_ent_idx,ent_cls_idx

# Sort index
def sort_and_save_index(cls_ent_idx,ent_cls_idx):
    sorted_cls_ent_idx = {}
    sorted_ent_cls_idx = {}
    for cls in cls_ent_idx.items():
        sorted_cls_ent_idx[cls[0]] = dict(sorted(cls[1].items(), key=lambda item: item[1],reverse=True))
    for ent in ent_cls_idx.items():
        sorted_ent_cls_idx[ent[0]] = dict(sorted(ent[1].items(), key=lambda item: item[1],reverse=True))

    # Save index
    for filename_output, index_dict in [('cls_ent_idx.json', sorted_cls_ent_idx),
                                        ('ent_cls_idx.json', sorted_ent_cls_idx)]:  # ,
        # cls and ent deliberately interchanged
        # ('ent_qid_idx.json',linked_cls_ent_idx)]:#,
        # ('cls_qid_idx.json',linked_ent_cls_idx)]:
        with open(filename_output, 'w') as f:
            json.dump(index_dict, f)

    return sorted_cls_ent_idx,sorted_ent_cls_idx

def load_index():
    with open("cls_ent_idx.json", 'r') as f:
        sorted_cls_ent_idx = json.load(f)
    with open("ent_cls_idx.json", 'r') as f:
        sorted_ent_cls_idx = json.load(f)
    return sorted_cls_ent_idx,sorted_ent_cls_idx

# Generate qids for top ten
def generate_qids(sorted_cls_ent_idx):
    # Save to index and table
    linked_cls_ent_idx = {}
    #linked_ent_cls_idx = {}
    linked_ent_table = pd.DataFrame(columns=['Keyword Entity Name','QID SPARQL', 'Score SPARQL' ,'QID Pywikibot','Score Pywikibot', 'QID Benchmark', 'Score Benchmark'])
    cls_count = 0
    ent_tracker = set()
    for cls in sorted_cls_ent_idx.items():
        if True:#cls_count < 10:
            linked_cls_ent_idx[cls[0]] = {}
            #print(cls[0])
            ent_count = 0
            for ent in cls[1].items():
                #print(ent[0])
                if True:#ent_count < 10:
                    name = ent[0]
                    # lemmatize name
                    #TODO: check if not twice
                    #name = WordNetLemmatizer().lemmatize(name)
                    # Retrieve QIDs
                    qid_sparql = get_qid_sparql(name)
                    qid_pywikibot = get_qid_pywikibot(name)
                    # Save to index
                    linked_cls_ent_idx[cls[0]][ent[0]] = qid_pywikibot#(qid_sparql,qid_pywikibot)
                    # Save to table
                    if name not in ent_tracker:
                        print(name)
                        new_row = {'Keyword Entity Name': name,'QID SPARQL': qid_sparql,'QID Pywikibot': qid_pywikibot}
                        linked_ent_table = linked_ent_table.append(new_row, ignore_index=True)
                        ent_tracker.add(name)
                ent_count += 1
        cls_count += 1

    # Save table
    linked_ent_table.to_csv(os.path.join('ent_qid_table.csv'), sep=';')
    return linked_cls_ent_idx

def predict_text_mscs(table,n_gram_lengths):

    # Create prediction table
    prediction_table = pd.DataFrame(columns=['de','mscs_actual','mscs_predicted','confidences','overlap_ratio'])

    # Open index
    sorted_cls_ent_idx,sorted_ent_cls_idx = load_index()

    # mscs actual vs. predicted
    mscs_actual = {}
    mscs_predicted = {}
    mscs_pred_conf = {}
    overlap_ratios = []

    sstopwords = get_stopwords()
    # predict mscs for each doc abs text
    tot_rows = len(table)
    latest_progress = 0
    for idx in range(tot_rows):
        #print(idx)
        current_progress = round(idx/tot_rows*100,1)
        if current_progress != latest_progress:
            print(current_progress,'%')
            latest_progress = current_progress
        text = table['text'][idx]
        mscs_actual[idx] = get_mscs(table,idx)
        mscs_predicted_stat = {}
        for n in n_gram_lengths:
            nngrams = ngrams(text.split(), n)
            try:
                for nngram in nngrams:
                    entity = ''
                    for word in nngram:
                        entity += word + ' '
                    entity = entity[:-1]
                    try:
                        if sorted_ent_cls_idx[entity] is not None and entity not in sstopwords:
                            #print(entity)
                            #mscs_predicted[idx].extend(list(sorted_ent_cls_idx[entity])[0:1])
                            for cls in sorted_ent_cls_idx[entity].items():
                                try:
                                    # SELECTION HERE
                                    mscs_predicted_stat[cls[0]] += 1#cls[1]#1 # weightedcontribution or binarycontribution
                                except:
                                    mscs_predicted_stat[cls[0]] = 1
                    except:
                        pass
            except:
                pass

        if len(mscs_predicted_stat) != 0:
            # sort
            sorted_mscs_predicted_stat = dict(
                    sorted(mscs_predicted_stat.items(), key=lambda item: item[1], reverse=True))

            # get (normalized) prediction (confidence)
            nr_mscs_cut_off = 1
            mscs_predicted[idx] = list(sorted_mscs_predicted_stat)[:nr_mscs_cut_off]#cut off at fixed nr_mscs_cut_off or dynamic number #len(mscs_actual[idx])
            mscs_cut_off = list(sorted_mscs_predicted_stat.items())[:nr_mscs_cut_off]
            # norm
            tot = sum(v for k,v in mscs_cut_off)
            # confidence
            mscs_pred_conf[idx] = [v/tot for k,v in mscs_cut_off]

            # compare mscs actual to predicted
            common_mscs = len(set(mscs_actual[idx]).intersection(set(mscs_predicted[idx])))
            total_mscs = nr_mscs_cut_off #len(mscs_predicted[idx])# + len(mscs_actual[idx])
            overlap_ratio = round(common_mscs/total_mscs,3)
            overlap_ratios.append(overlap_ratio)
            #print(idx,overlap_ratio)

            # extend prediction table
            new_row = {'de': table['de'][idx],'mscs_actual': mscs_actual[idx],'mscs_predicted': mscs_predicted[idx],'confidences': mscs_pred_conf[idx],'overlap_ratio': overlap_ratio}
            prediction_table = prediction_table.append(new_row,ignore_index=True)
            pass

    # save prediction table
    print('save prediction table...')
    prediction_table.to_csv('mscs_prediction_table.csv')

    print(np.mean(overlap_ratios))

    return None

def train_test_split(table,train_split_rate):

    # get split
    total_docs = len(table)
    train_split_docs = int(total_docs * train_split_rate)
    nr_docs = train_split_docs
    cls_ent_idx, ent_cls_idx = generate_msc_keyword_mapping(table, nr_docs)

    # save train-test split
    print('save train-test split...')
    with open('cls_ent_idx_split.json', 'w') as f:
        json.dump(cls_ent_idx, f)
    with open('ent_cls_idx_split.json', 'w') as f:
        json.dump(ent_cls_idx, f)

def get_sparse_mscs(table):
    # Create msc frequency index
    msc_freq_idx = {}

    # Iterate documents to get index
    tot_rows = len(table)
    for idx in range(tot_rows):
        mscs = table['MSC'][idx].split()
        for msc in mscs:
            try:
                msc_freq_idx[msc] += 1
            except:
                msc_freq_idx[msc] = 1

    # Get sparse mscs
    for msc in msc_freq_idx.items():
        msc_name = msc[0]
        msc_freq = msc[1]
        if msc_freq < 20:
            print(msc_name)
    # 1003 results for < 10
    # 1507 results for < 10

def predict_mscs(ent_cls_dict):
    # get confidences
    for ent in ent_cls_dict.items():
        ent_key = ent[0]
        ent_val = ent[1]
        total = sum(ent_val.values(), 0.0)
        for msc in ent_val.items():
            msc_key = msc[0]
            msc_val = msc[1]
            ent_cls_dict[ent_key][msc_key] /= total

def print_dataset_statistics(sorted_cls_ent_idx,sorted_ent_cls_idx):

    # average length
    def get_mean_count(idx):
        lengths = []
        for item in idx.items():
            lengths.append(len(item[1]))
        avg_length = np.mean(lengths)
        return avg_length
    cls_ent_idx_avg_count = get_mean_count(sorted_cls_ent_idx)
    print('Average entity per class count: ' + str(cls_ent_idx_avg_count))
    ent_cls_idx_avg_count = get_mean_count(sorted_ent_cls_idx)
    print('Average class per entity count: ' + str(ent_cls_idx_avg_count))

    # entropy
    def get_mean_entropy(idx):
        entropies = []
        for cls in idx.items():
            frequencies = [ent[1] for ent in cls[1].items()]
            entropies.append(scipy.stats.entropy(frequencies))
        avg_entropy = np.mean(entropies)
        return avg_entropy
    cls_ent_idx_avg_entropy = get_mean_entropy(sorted_cls_ent_idx)
    print('Average entity per class entropy: ' + str(cls_ent_idx_avg_entropy))
    ent_cls_idx_avg_entropy = get_mean_entropy(sorted_ent_cls_idx)
    print('Average class per entity entropy: ' + str(ent_cls_idx_avg_entropy))

    return 0

def load_data():
    # load data
    # msc(keyword) index
    with open(dict_path, 'r') as f:
        keyword_msc_index = json.load(f)
    # raw data
    raw_data = None #pd.read_csv(data_path)

    return raw_data,keyword_msc_index

def get_mrmscs_dict():

    try:
        with open('mrmscs_dict.json', 'r') as f:
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

        with open('mrmscs_dict.json','w') as f:
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

def compare_mr_keyword_refs_dcgs(table):

    # get eval table
    print('Load data')
    _, keyword_msc_index = load_data()
    raw_data = table
    mrmscs_dict = get_mrmscs_dict()
    print('Get DCGs')
    get_dcg_table(raw_data, mrmscs_dict, keyword_msc_index)

    # eval eval table
    eval_table = pd.read_csv(eval_path)
    compare_DCGs(eval_table)

###########
# EXECUTE #
###########

# Set paths
root_path = r'C:\Users\phili\Downloads'
file_name = 'out.csv'
data_path = os.path.join(root_path,file_name)
dict_path = 'ent_cls_idx_split.json'
eval_path = 'keywords_vs_refs_mrmscs.csv'
mrms_path = 'msc-mapping-zbmath-ams.csv'

# 0) Load input table
print('\nLoad input table...\n')
table = pd.read_csv(data_path,delimiter=',')

# Set parameter
tot_rows = len(table)
train_split_rate = 0.7
test_split = train_split_rate
nr_docs = int(tot_rows*train_split_rate)
nr_mscs_cutoff = 10

#1) Generate MSC-keyword mapping
print('\nGenerate MSC-keyword mapping...\n')
cls_ent_idx,ent_cls_idx = generate_msc_keyword_mapping(table,nr_docs)
sorted_cls_ent_idx,sorted_ent_cls_idx = sort_and_save_index(cls_ent_idx,ent_cls_idx)
#sorted_cls_ent_idx,sorted_ent_cls_idx = load_index()
#1*) Dataset statistics
print('\nDataset statistics:\n')
print_dataset_statistics(sorted_cls_ent_idx,sorted_ent_cls_idx)

#2) Predict MSCs
print('\nPredict MSCs...\n')
predict_text_mscs(table,n_gram_lengths=[2,3])

#3) Evaluate MSC predictions
print('\nEvaluate MSC predictions...\n')
train_test_split(table,train_split_rate)
get_sparse_mscs(table)

#4) Compare to MR-MSCs and References-MSCs
compare_mr_keyword_refs_dcgs(table)

print('end')
