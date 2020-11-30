# DRQA
Course project for Advanced NLP

## Document Retriever
docretriever.py: Use the DocumentRetriever class to retrieve Wikipedia articles. Pass the question as an arugemnt to the find() method.
```
ret = DocumentRetriever()
ret.find("Your question here?")
```
## Document Reader

samples.py: Takes json files of training and validation SQuAD sets and convert them to samples of each containing. 
          train contains: id,context tokens, context ent, context tags, context features, que, context span,ans start, ans end
          val contains: id,context tokens, context ent, context tags, context features, que, answers
drqareader.py: The RNN network for reader model suggested in DrQA. Implemented in pytorch. The network contains LSTM bi-directional Multi layer network.

fusionmodel.py: The RNN network for reader model of Fully aware multi-level fusion network. Implemented in pytorch. The network contains LSTM bi-directional Multi layer network.

srumodel.py: The RNN network for reader model of Fully aware multi-level fusion network. Implemented in pytorch. The network contains SRU bi-directional Multi layer network.  

train.py: This is responsible for training the dataset. Converts the samples into mini-batches and train on specified network.

interact.py: Reader interact model to predict the answer given a context.

test.py: Given a question as input, finds the context paragraph from wikipedia dataset and predicts the answer by calling interact.py
