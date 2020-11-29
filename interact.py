from samples import annotate
import torch
import pickle
from train import Train,__generate_batches
from samples import __generate_samples



def __get_vocab_set(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def predict(question,context,net_path,model,vocabfile):
    vocab_set = __get_vocab_set(vocabfile)
    embedding = torch.tensor(vocab_set['embeddings'])
    embedding = embedding.float()

    vocab_id_map = {w: i for i, w in enumerate(vocab_set['vocab'])}
    tag_id_map = {w: i for i, w in enumerate(vocab_set['tag_vocab'])}
    ent_id_map = {w: i for i, w in enumerate(vocab_set['ent_vocab'])}

    sample = ("dummy_id", context, question, "dummy", "dummy")
    sample = annotate(sample)

    sample = [sample, sample]
    samples = __generate_samples(sample, vocab_id_map, tag_id_map, ent_id_map)
    pickle.dump(samples, open("new_sample", 'wb'))

    config = {}
    config['vocab_size'] = embedding.size(0)
    config['e_dim'] = embedding.size(1)
    config['tag_size'] = len(vocab_set['tag_vocab'])
    config['ent_size'] = len(vocab_set['ent_vocab'])

    samples, _ = __generate_batches(samples, 2, config['tag_size'], config['ent_size'], 'eval')

    tr = Train(config, embedding, model)
    return tr.loadandpredict(samples, True, net_path)

