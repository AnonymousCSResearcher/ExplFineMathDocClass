import pandas as pd

root_path = ''#'evaluation/alldocs/ngrams_2-3/'

# get tables
files = ['mscs_prediction_table.csv'] #['mscs_prediction_table_binarycontribution.csv','mscs_prediction_table_weightedcontribution.csv']
tables = [pd.read_csv(root_path + file,delimiter=';') for file in files]

# get overlap ratios
overlap_ratios = []
for table in tables:
    for idx in table.iloc:
        overlap_ratios.append(idx['overlap_ratio'])

a = overlap_ratios
correctly_predicted_one_msc = len([i for i in a if i != 0])/len(a)#0.7249258734046667

print()