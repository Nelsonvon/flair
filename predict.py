from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from  flair.data_fetcher import NLPTaskDataFetcher
from flair.training_utils import Metric
import argparse
import os

def evaluate(model, test_set, dir=None, mt = None):
    columns = {0: 'text', 1: 'gold'}
    sentences_test = NLPTaskDataFetcher.read_column_data(test_set, columns)

    for sentence in sentences_test:
        sentence = sentence
        sentence.convert_tag_scheme(tag_type="gold", target_scheme='iobes')

    tagger = SequenceTagger.load_from_file(model)



    lines = []
    metric = Metric('Evaluation')

    result = tagger.predict(sentences_test)
    for sent in result:
        for tok in sent:
            predicted = tok.get_tag("ner").value
            gold = tok.get_tag("gold").value

            eval_line = "{}\t{}\t{}\n".format(tok.text, gold, predicted)

            if predicted != 'O':
                # true positives
                if predicted == gold:
                    metric.add_tp("ner")
                # false positive
                if predicted != gold:
                    metric.add_fp("ner")

            # negatives
            if predicted == 'O':
                # true negative
                if predicted == gold:
                    metric.add_tn("ner")
                # false negative
                if predicted != gold:
                    metric.add_fn("ner")

            lines.append(eval_line)

    #         if predicted==gold:
    #             metric.add_tp()
    #         if predicted == ""
        lines.append("\n")
    print(str(metric))

    if dir is not None:
        test_tsv = os.path.join(dir, "predicted_{}.conll".format(mt))
        with open(test_tsv, "w", encoding='utf-8') as outfile:
            outfile.write(''.join(lines))


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", default="/work/smt2/tokarchuk/flair_models/tac15_default_flair04/best-model.pt")
    argparser.add_argument("--ts", default="/work/wwang/corpora/TAC/no-duplicates/eng_tac_2016_eval.conll.merged")
    argparser.add_argument("--out_dir", default="/work/smt2/tokarchuk/flair_predictions/bytac15/")
    argparser.add_argument("--out_tag", default="tac16_dev")

    args = argparser.parse_args()
    evaluate(args.model,args.ts,args.out_dir, args.out_tag)

