path = "C:\\Users\\phili\\Downloads\\WikiDumps\\"
#path = "D:\\NTCIR-12_MathIR_arXiv_Corpus\\output_Explainability\\WikiDumps\\"
file = "enwiki-latest-all-titles-in-ns0"

def get_Wikipedia_article_names(n_gram_length):

    with open(path + file,"r",encoding="utf-8") as f:
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