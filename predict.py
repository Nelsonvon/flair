from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from  flair.data_fetcher import NLPTaskDataFetcher
from flair.training_utils import Metric
from pathlib import Path
import argparse
import os

def evaluate(model, test_set, dir=None, mt = None):
    columns = {0: 'text', 1: 'ner'}
    sentences_test = NLPTaskDataFetcher.read_column_data(test_set, columns)

    for sentence in sentences_test:
        sentence = sentence
        sentence.convert_tag_scheme(tag_type="ner", target_scheme='iobes')

    tagger = SequenceTagger.load_from_file(model)

    print("Testing using best model ...")
    metric, eval_loss = ModelTrainer.evaluate(tagger, sentences_test, eval_mini_batch_size=32,
                          embeddings_in_memory=True, out_path=Path(os.path.join(dir, "predicted_{}.conll".format(mt))))
    print(metric)


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", default="/work/smt2/tokarchuk/flair_models/tac15_default_flair04/best-model.pt")
    argparser.add_argument("--ts", default="/work/wwang/corpora/CoNLL/original/test.txt")
    argparser.add_argument("--out_dir", default="/work/smt2/tokarchuk/flair_predictions/conll")
    argparser.add_argument("--out_tag", default="tac16_dev")

    args = argparser.parse_args()
    evaluate(args.model,args.ts,args.out_dir, args.out_tag)

