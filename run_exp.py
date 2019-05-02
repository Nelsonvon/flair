import json
import datetime
import mimesis
import pandas as pd

base_conf = {
    "base_dir": "",
    "task": "TAC",
    "max_epochs": 150,
    "anneal_factor": 0.5,
    "hidden_size": 256,
    "char_embeddings": False,
    "filenames": {
        "train": [
            ""
        ],
        "dev": "",
        "test": ""
    },
    "save_model": True,
    "use_crf": True,
    "model_tag": "tac15_autotag_flair04",
    "embeddings_name": "glove",
    "word_embeddings": True,
    "rnn_layers": 1,
    "train_with_dev": False,
    "use_rnn": True,
    "learning_rate": 0.1,
    "tag_type": "ner",
    "mini_batch_size": 32,
    "charlm_embeddings": True,
    "model_dir": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/",
    "convert_tag": "ner",
    "inmem": True,
    "bert_embeddings": False,
    "bert_model": "bert-base-cased",
    "optimizer": "sgd"
}


"""
{
"filenames": {
        "dev": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_dev.conll",
        "test": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_test.conll",
        "train": [
            "/work/wwang/corpora/OntoNotes/training/ontonotes.train.rescored.th06.conll"
        ]
    },
'desc': "ontonotes train rescored th 0.6"
},
"""
"""
{
"filenames": {
        "dev": "/work/wwang/corpora/CoNLL/original/valid.txt",
        "test": "/work/wwang/corpora/CoNLL/original/test.txt",
        "train": [
            "/work/wwang/corpora/CoNLL/training/train.rescored.fixed.099.conll"
        ]
    },
'desc': "CoNLL train rescored fixed th 0.99"
},
"""
to_change = \
    [
{
"filenames": {
        "dev": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_dev.conll",
        "test": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_test.conll",
        "train": [
            "/work/wwang/corpora/OntoNotes/training/ontonotes.train.rescored.fixed.th099.conll"
        ]
    },
'desc': "ontonotes train rescored fixed th 0.99"
},
{
"filenames": {
        "dev": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_dev.conll",
        "test": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_test.conll",
        "train": [
            "/work/wwang/corpora/OntoNotes/training/ontonotes.train.rescored.fixed.th098.conll"
        ]
    },
'desc': "ontonotes train rescored fixed th 0.98"
},
{
"filenames": {
        "dev": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_dev.conll",
        "test": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_test.conll",
        "train": [
            "/work/wwang/corpora/OntoNotes/training/ontonotes.train.rescored.fixed.th095.conll"
        ]
    },
'desc': "ontonotes train rescored fixed th 0.95"
},
{
"filenames": {
        "dev": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_dev.conll",
        "test": "/work/wwang/corpora/OntoNotes/original-processed-CoNLL/ontonotes_test.conll",
        "train": [
            "/work/wwang/corpora/OntoNotes/training/ontonotes.train.rescored.fixed.th09.conll"
        ]
    },
'desc': "ontonotes train rescored fixed th 0.9"
},

]
commands = []
experiments = []
for datasets in to_change:
    new_config = dict(base_conf)
    now = datetime.datetime.now().strftime("%Y_%m_%d")
    model_tag = "flair04_{}_{}".format(now, mimesis.Cryptographic().token_hex(5))
    new_config["model_tag"] = model_tag
    for k, v in datasets.items():
        new_config[k] = v

    config_name = "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_configs/{}".format(model_tag)
    with open(config_name, "w") as cf:
        json.dump(new_config, cf, indent=4, sort_keys=True)

    job_name = "{}".format(model_tag)
    cluster_command = "echo \"/u/qfeng/Project/work_37/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/train.py --config {}\" " \
                      "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l gpu=1 -l h_vmem=40G -l qname=*1080* -l h_rt=60:00:00 ".format(
        config_name, job_name)

    commands.append(cluster_command)
    experiments.append({"name": model_tag, "date": now, "config": config_name,
                        "train": "\n".join(new_config["filenames"]["train"]),
                        "dev": new_config["filenames"]["dev"],
                        "test": new_config["filenames"]["test"],
                        "checked": 0,
                        "job_run": cluster_command,
                        "description": datasets["desc"]
                        })

with open("/work/smt2/qfeng/Project/auto_dataset_foundation/run/run_experiments_{}.sh".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
          "w") as exp:
    for c in commands:
        exp.write(c)
        exp.write("\n")

result = pd.DataFrame(experiments)
result.to_csv("/work/smt2/qfeng/Project/auto_dataset_foundation/logs/experiments/experiments_{}.csv".format(now), mode="a")
