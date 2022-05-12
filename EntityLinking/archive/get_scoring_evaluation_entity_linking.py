import os
import pandas as pd

# set file path
folder_path = r"C:\Users\phili\Dropbox\PhD\Projects\Formula Clustering\FormulaFeatureAnalysis\docsecabsTextMathClassClust\Entities&Categories\EntityLinking\documents\with_msc\[0705.3017] Vortex in axion condensate as a dark matter halo\Abstract text\2grams"
file_name = "unlemmatiz.csv"
file_path = os.path.join(folder_path,file_name)

# open table
table = pd.read_csv(file_path,delimiter=",")

# evaluate

# get column names and row number
col_names = list(table.columns)
nr_rows = table.shape[0]
# track eval scores
eval_scores = {}
eval_classes = ['TP','FP','FN','TN']
# iterate columns
for row in table.iterrows():
    row_index = row[0]
    row_content = row[1]

    # get manual classified relevance
    Relevance = row_content['Relevance']

    # get binary relevance classification (TP,FP,FN,TN)
    offset = 4
    for i in range(1,7):

        # get desired column names and content
        content_col_name = col_names[offset+2*(i-1)]
        content_col_content = row_content[content_col_name]
        eval_col_name = "eval" + str(i)

        # calculate rel class
        if Relevance != "-" and content_col_content != "-":
            eval_col_content = eval_classes[0] # "TP" # true positive
        elif Relevance == "-" and content_col_content != "-":
            eval_col_content = eval_classes[1] # "FP" # false positive
        elif Relevance != "-" and content_col_content == "-":
            eval_col_content = eval_classes[2] # "FN" # false negative
        elif Relevance == "-" and content_col_content == "-":
            eval_col_content = eval_classes[3] # "TN" # true negative

        # write scores to eval_scores dict
        try:
            eval_scores[eval_col_name][eval_col_content] += 1
        except:
            try:
                eval_scores[eval_col_name][eval_col_content] = 1
            except:
                eval_scores[eval_col_name] = {}
                eval_scores[eval_col_name][eval_col_content] = 1

        # write rel class to table
        table.iloc[row_index,table.columns.get_loc(eval_col_name)] = eval_col_content

    pass

# calculate precision, recall, and F1 measure
def set_missing_classes_to_zero(eval_mode):
    for eval_class in eval_classes:
        if eval_mode.get(eval_class) is None:
            eval_mode[eval_class] = 0
def get_precision(eval_mode):
    return round(eval_mode['TP']/(eval_mode['TP'] + eval_mode['FP']),2)
def get_recall(eval_mode):
    return round(eval_mode['TP']/(eval_mode['TP'] + eval_mode['FN']),2)
def get_F1_measure(precision,recall):
    return round(2*(precision*recall)/(precision+recall),2)

for eval_mode in eval_scores.items():
    eval_mode_name = eval_mode[0]
    eval_mode_content = eval_mode[1]

    # get precision, recall, and F1 measure
    set_missing_classes_to_zero(eval_mode_content)
    precision = get_precision(eval_mode_content)
    recall = get_recall(eval_mode_content)
    F1_measure = get_F1_measure(precision,recall)

    # save to eval_scores dict
    eval_scores[eval_mode_name]['precision'] = precision
    eval_scores[eval_mode_name]['recall'] = recall
    eval_scores[eval_mode_name]['F1_measure'] = F1_measure

    # write to table
    table.iloc[nr_rows-3, table.columns.get_loc(eval_mode_name)] = precision
    table.iloc[nr_rows-2, table.columns.get_loc(eval_mode_name)] = recall
    table.iloc[nr_rows-1, table.columns.get_loc(eval_mode_name)] = F1_measure

# save table
table.to_csv(file_path.strip(".csv") + "_sco" + ".csv",index=False)

print("end")
