import json

# get index
ent_cls_dict_path = 'evaluation/alldocs/ngrams_2-3/ent_cls_idx.json'

with open(ent_cls_dict_path,'r') as f:
    ent_cls_dict = json.load(f)

# get confidences
for ent in ent_cls_dict.items():
    ent_key = ent[0]
    ent_val = ent[1]
    total = sum(ent_val.values(), 0.0)
    for msc in ent_val.items():
        msc_key = msc[0]
        msc_val = msc[1]
        ent_cls_dict[ent_key][msc_key] /= total

print()