from os import listdir
import json
from bs4 import BeautifulSoup
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

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

# define cleaning
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
def nlp_clean(raw_str):

    lowered_str = raw_str.lower()
    tokenized_str = tokenizer.tokenize(lowered_str)
    swremoved = list(set(tokenized_str).difference(stopword_set))
    cleaned = []
    for word in swremoved:
        contains_digit = False
        for char in word:
            if char.isdigit():
                contains_digit = True
        if not contains_digit and len(word) > 3:
            cleaned.append(word)
    return cleaned

# exclude formulae, stopwords, html and letters from candidates
excluded = [">", "<", "=", "~",'"', "_"]
with open("../../stopwords.txt") as f:
    stopwords = [line.strip() for line in f]
#invalid = ["times"]
with open("../../letters.txt") as f:
    letters = [line.strip() for line in f]

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

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
                        docText = nlp_clean(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read())

                        # retrieve math data (formulae) from document
                        with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
                            filestring = f.read()
                            formulae = BeautifulSoup(filestring, 'html.parser').find_all('formula')

                        docMath = []
                        for formula in formulae:

                            formulaString = str(formula.contents)

                            # extract TeX formula
                            # formulaString
                            s = str(formula.contents)
                            start = 'alttext="'
                            end = '" display='
                            try:
                                #TeX = re.search('%s(.*)%s' % (start, end), s).group(1)
                                TeX = formula.contents[0].attrs['alttext']
                            except:
                                TeX = ""

                            # extract surrounding tex
                            index = filestring.find('alttext="' + TeX + '" display=')
                            surrounding_text_candidates = filestring[index - 500:index + 500]

                            for word in surrounding_text_candidates.split():
                                # lowercase and remove .,-()
                                word = word.lower()
                                char_excl = [".", ":", ",", "-", "(", ")",'=']
                                for c in char_excl:
                                    word = word.replace(c, "")
                                # not part of a formula environment
                                not_formula = not True in [ex in word for ex in excluded]
                                # not stopword
                                not_stopword = word not in stopwords
                                # not invalid html
                                #not_invalid = not True in [inv in word for inv in invalid]
                                # not a latin or greek letter
                                not_letter = word not in letters
                                if not_formula and not_stopword and not_letter: #and not_invalid
                                    #if TeX != "" and TeX not in stopwords and TeX not in letters:
                                    # check if around equation
                                    #if '=' in TeX:
                                    if len(TeX) > 10:
                                        docMath.append(word)

                        # augment indices
                        for word in docMath:#docText, docMath
                            # augment class_entity_index
                            try:
                                class_entity_index[classLab][word] += 1
                            except:
                                try:
                                    class_entity_index[classLab][word] = 1
                                except:
                                    class_entity_index[classLab] = {}
                                    class_entity_index[classLab][word] = 1
                            # augment entity_class_index
                            try:
                                entity_class_index[word][classLab] += 1
                            except:
                                try:
                                    entity_class_index[word][classLab] = 1
                                except:
                                    entity_class_index[word] = {}
                                    entity_class_index[word][classLab] = 1

# sort class_entity and entity_class index
sorted_class_entity_index = {}
for cls in class_entity_index.items():
    sorted_class_entity_index[cls[0]] = dict(sorted(cls[1].items(), key=lambda item: item[1],reverse=True))
sorted_entity_class_index = {}
for ent in entity_class_index.items():
    sorted_entity_class_index[ent[0]] = dict(sorted(ent[1].items(), key=lambda item: item[1],reverse=True))

# save class_entity and entity_class index
with open(outputPath + "sorted_class_entity_index_math.json","w") as f:
    json.dump(sorted_class_entity_index,f)
with open(outputPath + "sorted_entity_class_index_math.json","w") as f:
    json.dump(sorted_entity_class_index,f)

print("end")