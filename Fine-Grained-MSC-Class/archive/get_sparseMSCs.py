import pandas as pd
import os
import json

# Load table
filepath = r'C:\Users\phili\Downloads'
#filepath = r'C:\Users\Lenovo 17\Downloads'
filename_input = 'full.csv'
fullpath = os.path.join(filepath,filename_input)
table = pd.read_csv(fullpath,delimiter=',')

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

print('end')
