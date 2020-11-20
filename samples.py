import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
import json
import sys
import string
import spacy
import collections
import pickle
import unicodedata
import time
import msgpack
from memory_profiler import profile

stop_words = stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
nlp = spacy.load('en')
infile = open('/Users/skosgi/Downloads/nlp/Project/glove','rb')
glove = pickle.load(infile)
infile.close()

exp1 = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.DOTALL)
exp2 = re.compile(r'{\|(.*?)\|}',re.DOTALL)
exp3 = re.compile(r'\[\[category:(.*?)\]\]',re.DOTALL)
exp4 = re.compile(r'{{infobox((.|\n)*?)}}',re.DOTALL)

exp5 = re.compile(r'{{v?cite(.*?)}}',re.DOTALL)
exp6 = re.compile(r'\[\[file:(.*?)\]\]',re.DOTALL)
exp7 = re.compile(r'\{\{.*?\}\}',re.DOTALL)
exp8 = re.compile(r'\[\[Category.*?\]\]',re.DOTALL)
exp9 = re.compile(r'\=.*?\=')


def file2text(file):
    infile = open(file, 'r')
    text = infile.read()
    #text = "{{abcd}}abcdjhbfjrb"
    text = exp1.sub('', text)
    text = exp2.sub('', text)
    text = exp5.sub('', text)
    text = exp6.sub('', text)
    text = exp7.sub('',text)
    text = exp8.sub('',text)
    text = exp9.sub('',text)
    return text

def load_vocab():
    vocab = set()
    file = open("gensim_glove_vectors.txt", "r")
    for line in file:
        vocab.add(line.rstrip().split(" ")[0])

    outfile = open("glove_vocab", "wb")
    pickle.dump(vocab, outfile)



def tolist(df,type):
    samples = []
    if type == 'train':
        for i in df.index:
            samples.append((df['index'][i],df['context'][i],df['question'][i],df['text'][i],df['answer_start'][i],df['answer_start'][i]+len(str(df['text'][i]).rstrip())))
        return samples
    else:
        for i in df.index:
            answers = [w['text'] for w in df['answers'][i]]
            samples.append((df['id'][i],df['context'][i],df['question'][i],answers))
        return samples
def __removeHtmlTags(sequence):
    #########################################################################################
    # This method removes any HTML tags and gets only the text from the given sequence.
    #   For e.g., in the sequence "<H1>This is the header</H1>", it removes H1 tag
    #   and returns "This is the header".
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub(r'<(.*?)>', '', sequence)
        return sequence  # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing HTML tags in the sequence '{0}'. Error is: {1}; {2}".format(sequence,
                                                                                                         str(
                                                                                                             exc_type),
                                                                                                         str(
                                                                                                             exc_value))
        raise Exception(err)

def __replaceurls(sequence):
    #########################################################################################
    # This method removes urls in the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          sequence, flags=re.MULTILINE)
        return sequence  # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing HTML tags in the sequence '{0}'. Error is: {1}; {2}".format(sequence,
                                                                                                         str(
                                                                                                             exc_type),
                                                                                                         str(
                                                                                                             exc_value))
        raise Exception(err)

def __removePunctuations(sequence):
    #########################################################################################
    # This method removes any punctuations and gets only the text from the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub('[^A-Za-z0-9]+', ' ', sequence)
        return sequence  # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing punctuations in the sequence '{0}'. Error is: {1}; {2}".format(
            sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __normalizeText(text):
    return unicodedata.normalize('NFD', text)

def annotate(sample):
    _id, context, question = sample[0], sample[1], sample[2]

    c_doc = nlp(context)
    q_doc = nlp(question)
    context_tokens = [__normalizeText(token.text).lower() for token in c_doc]
    que_tokens = [__normalizeText(token.text).lower() for token in q_doc]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [word.tag_ for word in c_doc]
    que_tags = [word.tag_ for word in q_doc]
    context_ent = [word.ent_type_ for word in c_doc]
    que_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in q_doc}
    que_tokens_set = set(que_tokens)
    exact_match = [word in que_tokens_set for word in context_tokens]
    lemma_match = [(word.lemma_ if word.lemma_ != '-PRON-' else word.text.lower()) in que_lemma for word in c_doc]
    counter = collections.Counter(context_tokens)

    total_tokens = len(context_tokens)

    context_tf = [counter[word]/total_tokens for word in context_tokens]
    context_features = [exact_match,lemma_match,context_tf]

    return (_id,context_tokens,context_ent,context_features,context_tags,que_tokens,context,context_token_span)+sample[3:]

def corpusvocab(questions,context,vocab):

    q_counter = collections.Counter(word for d in questions for word in d)
    c_counter = collections.Counter(word for d in context for word in d)
    counter = q_counter+c_counter
    q_vocab = [word for word in q_counter if word in vocab]
    c_vocab = [word for word in c_counter.keys()-q_counter.keys() if word in vocab]
    sorted_vocab = sorted(q_vocab,key=q_counter.get,reverse=True)
    sorted_vocab += sorted(c_vocab,key=counter.get,reverse=True)
    sorted_vocab.insert(0,"<PAD>")
    sorted_vocab.insert(1,"<UNK>")
    return sorted_vocab,counter


def __squad_json_to_dataframe_train(input_file_path,
                                  record_path=['data', 'paragraphs', 'qas', 'answers'],
                                  verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1,
                     sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")

    return tolist(main,'train')


def __squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    #     ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    #     js['q_idx'] = ndx
    main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return tolist(main,'test')



def vocabulary(path):
    with open(path, "rb") as vocabfile:
        vocab = pickle.load(vocabfile)
    return vocab

def __generate_samples(squad,vocab_id_map,tag_id_map,ent_id_map):
    samples = []
    for sample in squad:
        q = sample[5]
        q_ids = [vocab_id_map[word] if word in vocab_id_map else 1 for word in q]

        c = sample[1]
        c_ids = [vocab_id_map[word] if word in vocab_id_map else 1 for word in c]

        c_tags = sample[4]
        c_tag_ids = [tag_id_map[word] for word in c_tags]

        c_ent = sample[2]
        c_ent_ids = [ent_id_map[word] for word in c_ent]

        samples.append((sample[0],c_ids,c_ent_ids,c_tag_ids,sample[3],q_ids)+sample[6:])
    return samples

if __name__ == '__main__':
    starttime = time.time()
    embedding_path = "/Users/skosgi/Downloads/nlp/Project/glove_vocab"
    e_vocab = vocabulary(embedding_path)
    print(len(e_vocab))

    embedding_dim = 300
    squad = __squad_json_to_dataframe_train('/Users/skosgi/Downloads/nlp/Project/train-v1.1.json')
    squad_dev = __squad_json_to_dataframe_dev('/Users/skosgi/Downloads/nlp/Project/dev-v1.1.json')
    for i in range(len(squad_dev)):
        annotations = annotate(squad_dev[i])
        squad_dev[i] = annotations
    for i in range(len(squad)):
        annotations = annotate(squad[i])
        squad[i] = annotations
        context_span = squad[i][-4]
        starts, ends = zip(*context_span)
        answer_start = squad[i][-2]
        answer_end = squad[i][-1]
        try:
            squad[i] = squad[i][:-3] + (starts.index(answer_start),ends.index(answer_end))
        except:
            squad[i] = squad[i][:-3] +(None,None)
    print("annotations and indexing answer is done")
    print("Number of train samples:{} and Number of test samples:{}".format(len(squad),len(squad_dev)))
    total = squad+squad_dev
    questions = [sample[5] for sample in total]
    contexts = [sample[2] for sample in total]

    sorted_vocab,counter = corpusvocab(questions, contexts, e_vocab)

    tag_counter = collections.Counter(word for sample in total for word in sample[4])
    tag_vocab = sorted(tag_counter,key=tag_counter.get,reverse=True)
    ent_counter = collections.Counter(word for sample in total for word in sample[2])
    ent_vocab = sorted(ent_counter,key=ent_counter.get,reverse=True)

    vocab_id_map = {word:i for i,word in enumerate(sorted_vocab)}
    tag_id_map = {word:i for i,word in enumerate(tag_vocab)}
    ent_id_map = {word:i for i,word in enumerate(ent_vocab)}

    train_samples = __generate_samples(squad,vocab_id_map,tag_id_map,ent_id_map)
    val_samples = __generate_samples(squad_dev,vocab_id_map,tag_id_map,ent_id_map)
    vocab_size = len(sorted_vocab)
    print(vocab_size)
    embeddings = np.zeros((vocab_size,embedding_dim))
    file = open("/Users/skosgi/Downloads/nlp/Project/gensim_glove_vectors.txt", "r")
    for line in file:
        embeds = line.rstrip().split(" ")
        word = __normalizeText(embeds[0])
        if word in vocab_id_map:
            embeddings[vocab_id_map[word]] = embeds[1:]

    vocab_set = {'vocab':sorted_vocab,
                 'tag_vocab':tag_vocab,
                 'ent_vocab':ent_vocab,
                 'embeddings':embeddings
                }
    with open("vocab_set","wb") as f:
        pickle.dump(vocab_set,f)

    #########################################################################################
    # train contains: id,context tokens, context ent, context tags, context features, que, context span,ans start, ans end
    # val contains: id,context tokens, context ent, context tags, context features, que, answers
    #########################################################################################
    samples = {'train':train_samples,
               'val':val_samples}
    with open("train_samples","wb") as f:
        pickle.dump(samples,f)

    print("Time taken for preprocess is:",time.time()-starttime)




