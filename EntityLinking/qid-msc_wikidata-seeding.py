import json
import pandas as pd

# Open keyword-QIDs table

keywords_qids_filename = 'zbmath keywords evaluation_all.csv'
keywords_qids = pd.read_csv(keywords_qids_filename,delimiter=',')

# Open keywords-MSCs index

keywords_mscs_filename = 'ent_cls_idx.json'
with open(keywords_mscs_filename,'r') as f:
    keywords_mscs = json.load(f)

# map qid to msc
header = 'qid,P3285,#\n'
comment_string = '"zbMATH Open keyword"'
example_line = 'Q74304,62J02,"zbMATH Open keyword"'
csv_string = header
for index, row in keywords_qids.iterrows():
    keyword = row['Keyword Entity Name']
    QID = row['QID Benchmark']
    if str(QID) != 'nan':
        try:
            MSC = list(keywords_mscs[keyword])[0]
            MSC_string = '"""' + MSC + '"""'
            csv_string += QID + ',' + MSC_string + ',' + comment_string + '\n'
        except:
            pass

# save to csv

qids_mscs_filename = 'qid-msc_wikidata-seeding.csv'
with open(qids_mscs_filename,'w') as f:
    f.write(csv_string)

print()