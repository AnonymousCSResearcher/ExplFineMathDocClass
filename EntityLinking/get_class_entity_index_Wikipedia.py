from os import listdir
import json
from bs4 import BeautifulSoup
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from EntityLinking.WikiDumps import get_Wikipedia_article_names

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'

datasetPath = basePath + "NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = basePath + "output_Explainability\\"

# Create class_entity and entity_class index
class_entity_index = {}
entity_class_index = {}

# Define class counter and desired classes
classCounter = {}
classLimit = 10
#desired_classes = ['astro-ph']
desired_classes = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics']
# Set mode (text or math)
mode = 'text'
# Set ngram length
n_gram_length = 1

# Load Wikipedia names
names = get_Wikipedia_article_names(n_gram_length)

# Define cleaning
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
def nlp_clean(raw_str):

    lowered_str = raw_str.lower()
    lowered_str = lowered_str.replace(".","").replace(",","")
    tokenized = set(tokenizer.tokenize(lowered_str))
    swremoved = tokenized.difference(stopword_set)
    cleaned_str = ""
    for word in swremoved:
        contains_digit = False
        for char in word:
            if char.isdigit():
                contains_digit = True
        if not contains_digit and len(word) > 3:
            cleaned_str += word + " "
    return cleaned_str[:-1]

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

# retrieve text data from document
def get_docText(datasetPath,Dir,File):
    text = nlp_clean(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read())

    docText = set()
    # Check for Wikpedia article names
    # USING LIST
    # for name in names:
    #     if name in text:
    #         docText.add(name)
    # USING DICT
    nngrams = ngrams(text.split(),n=n_gram_length)
    for nngram in nngrams:
        name = ''
        for word in nngram:
            name += word + " "
        try:
            docText.add(names[name[:-1]])
        except:
            pass

    return docText

# retrieve math data (formulae) from document
def get_docMath(datasetPath,Dir,File):
    with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
        filestring = f.read()
        formulae = BeautifulSoup(filestring, 'html.parser').find_all('formula')

    docMath = set()
    for formula in formulae:

        # extract TeX formula
        # formulaString
        try:
            TeX = formula.contents[0].attrs['alttext']
        except:
            TeX = ""

        # extract surrounding tex
        index = filestring.find('alttext="' + TeX + '" display=')
        surrounding_text_candidates = filestring[index - 500:index + 500]
        text = nlp_clean(surrounding_text_candidates)

        # Check for Wikpedia article names
        # USING LIST
        # for name in names:
        #     if name in surrounding_text_candidates:
        #         docMath.append(name)
        # USING DICT
        #nngrams = text.split()
        nngrams = ngrams(text.split(),n=n_gram_length)
        for nngram in nngrams:
            name = ''
            # name = nngram + " "
            for word in nngram:
                name += word + " "
            try:
                docMath.add(names[name[:-1]])
            except:
                pass

    return docMath

# Fetch content and labels of documents
for Dir in listdir(datasetPath):
    for prefix in valid_folder_prefix:
        if Dir.startswith(prefix):
            for File in listdir(datasetPath + "\\" + Dir):
                if not File.startswith("1") and File.endswith(".tei"):
                    # fetch label from file prefix
                    if Dir.startswith("9"):
                        classLab = File.split("9")[0]
                    else:
                        classLab = File.split("0")[0]
                    # check if class is desired and limit is not exceeded
                    try:
                        classCounter[classLab] += 1
                    except:
                        classCounter[classLab] = 1
                    #if True: # switch off desired_classes / classLimit constraints
                    if classLab in desired_classes and classCounter[classLab] <= classLimit:
                        print(Dir + "\\" + File)

                        # retrieve text data from document
                        if mode == 'text':
                            docNames = get_docText(datasetPath, Dir, File)

                        # retrieve math data (formulae) from document
                        if mode == 'math':
                            docNames = get_docMath(datasetPath, Dir, File)

                        # augment indices
                        for name in docNames: #docText or docMath
                            # augment class_entity_index
                            try:
                                class_entity_index[classLab][name] += 1
                            except:
                                try:
                                    class_entity_index[classLab][name] = 1
                                except:
                                    class_entity_index[classLab] = {}
                                    class_entity_index[classLab][name] = 1
                            # augment entity_class_index
                            try:
                                entity_class_index[name][classLab] += 1
                            except:
                                try:
                                    entity_class_index[name][classLab] = 1
                                except:
                                    entity_class_index[name] = {}
                                    entity_class_index[name][classLab] = 1

# sort class_entity and entity_class index
sorted_class_entity_index = {}
for cls in class_entity_index.items():
    sorted_class_entity_index[cls[0]] = dict(sorted(cls[1].items(), key=lambda item: item[1],reverse=True))
sorted_entity_class_index = {}
for ent in entity_class_index.items():
    sorted_entity_class_index[ent[0]] = dict(sorted(ent[1].items(), key=lambda item: item[1],reverse=True))

# save class_entity and entity_class index
with open(outputPath + "most_frequent_" + mode + "_class_entity.json","w") as f:
    json.dump(sorted_class_entity_index,f)
with open(outputPath + "most_frequent_" + mode + "_entity_class.json","w") as f:
    json.dump(sorted_entity_class_index,f)

print("end")