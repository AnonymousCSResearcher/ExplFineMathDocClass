import json
from scipy import stats
import numpy as np

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'
outputPath = basePath + 'output_Explainability\\'

# Set mode (text or math)
mode = 'math'

# load class_entity and entity_class index
with open(outputPath + "sorted_class_entity_index_" + mode + ".json","r") as f:
    sorted_class_entity_index = json.load(f)
with open(outputPath + "sorted_entity_class_index_" + mode + ".json","r") as f:
    sorted_entity_class_index = json.load(f)

entropies = []
for cls in sorted_entity_class_index.items():
    # (shannon) entropy
    entropy = stats.entropy([entry[1] for entry in cls[1].items()])
    entropies.append(entropy)

print(np.mean(entropies))