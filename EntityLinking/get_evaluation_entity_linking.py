import pywikibot
import SPARQLWrapper
import time
from EntityLinking.WikiDumps import get_Wikipedia_article_names
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk import WordNetLemmatizer
from nltk import ngrams

# FUNCTIONS

# nlp

def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw

def stemming_lemmatization(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def nlp_clean(text):
    tokens_without_sw = remove_stopwords(text)
    clean_words = []
    for word in tokens_without_sw:
        clean_words.append(stemming_lemmatization(word))
    return ' '.join(clean_words)

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
        qid = None
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

    try:
        qid = sparql_results['results']['bindings'][0]['item']['value'].split("/")[-1]
    except:
        qid = None

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

def get_evaluation_nlp_entity_linking(filepath,filename,n_gram_length):

    # open file
    with open(filepath + filename, 'r') as f:
        text = f.read()

    # remove <math> formula tags

    # remove stopwords and punctuation
    with open("stopwords.txt",'r') as f:
        stopwords = f.readlines()
    punctuations = [',',';','.','!','?']
    for stopword in stopwords:
        text = text.replace(stopword,"")
    for punctuation in punctuations:
        text = text.replace(punctuation,"")

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

    # save csv
    with open(filepath + filename + "_" + str(n_gram_length) + "grams(cleaned).csv",'w') as f:
        f.writelines(csv_lines)

def get_evaluation_mathir_entity_linking():
    print()

# EXECUTE

# predict entity links

#filepath = "documents/test/"
#filename = "Hawking Evaporation and Mutual Information Optimization (Abstract text).txt"
#filename = "The Cosmological Slingshot Scenario in details (Abstract text).txt"

filepath = "documents/with_msc/"
#filename = "[0704.0221] The Return of a Static Universe and the End of Cosmology (Abstract).txt"
filename = "[0705.3017] Vortex in axion condensate as a dark matter halo (Abstract text).txt"
#filename = "[0704.1436] Symmetries and the cosmological constant (Abstract text).txt"

get_evaluation_nlp_entity_linking(filepath=filepath,filename=filename,n_gram_length=2)

#NEXT:
#https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python
#https://paperswithcode.com/task/entity-linking/latest
#https://paperswithcode.com/paper/cholan-a-modular-approach-for-neural-entity

print("end")