import argparse
import json
import os
from typing import List
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings, CharacterEmbeddings, \
    BertEmbeddings
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger


def train(params):
    # 1. get the corpus
    print(NLPTask[params["task"]])
    corpus = NLPTaskDataFetcher.load_corpus(task=NLPTask[params["task"]], files=params['filenames'])
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    embedding_types = []

    if params["word_embeddings"] is True:
        embedding_types.append(WordEmbeddings(params["embeddings_name"]))

    if params["bert_embeddings"] is True:
        embedding_types.append(BertEmbeddings(params["bert_model"]))

    if params["char_embeddings"] is True:
        # comment in this line to use character embeddings
        embedding_types.append(CharacterEmbeddings())

    if params["charlm_embeddings"] is True:
        embedding_types.append(PooledFlairEmbeddings('news-forward', pooling='min'))
        embedding_types.append(PooledFlairEmbeddings('news-backward', pooling='min'))

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger

    tagger = SequenceTagger(hidden_size=params["hidden_size"],
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_rnn=params["use_rnn"],
                            rnn_layers=params["rnn_layers"],
                            use_crf=params["use_crf"]
                            )

    base_path = os.path.join(params["model_dir"], params["model_tag"])
    os.makedirs(base_path, exist_ok=True)

    with open(os.path.join(base_path, 'config.json'), "w") as cfg:
        json.dump(params, cfg)
    # initialize trainer

    if "optimizer" in params:
        if params["optimizer"] == "adam":
            optim = Adam
        elif params["optimizer"] == "sgd":
            optim = SGD
    else:
        optim = SGD
    trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=optim)

    trainer.train(base_path, EvaluationMetric.MICRO_F1_SCORE, mini_batch_size=params["mini_batch_size"],
                  max_epochs=params["max_epochs"], save_final_model=params["save_model"],
                  train_with_dev=params["train_with_dev"], anneal_factor=params["anneal_factor"],
                  embeddings_in_memory=params["inmem"], test_mode=False)

    plotter = Plotter()
    plotter.plot_training_curves(os.path.join(base_path, "loss.tsv"))
    plotter.plot_weights(os.path.join(base_path, 'weights.txt'))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="config.json")

    args = arg_parser.parse_args()

    with open(args.config) as cfg:
        params = json.load(cfg)

    print(params)
    train(params)
