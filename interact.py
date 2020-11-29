from samples import annotate
import torch
import pickle
from train import Train,__generate_batches
from samples import __generate_samples



question = "Which NFL team represented the AFC at Super Bowl 50?"
context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."

pathfile = "/Users/skosgi/Downloads/network26"


def __get_vocab_set(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def predict(question,context,net_path,model):
    vocab_set = __get_vocab_set("vocab_set")
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
    print(tr.loadandpredict(samples, True, pathfile))

