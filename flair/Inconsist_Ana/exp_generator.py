import json
import datetime
import mimesis
import pandas as pd

base_conf = {
    "dataset": 'CoNLL',
	"sent_sim_th": 0.1,
	"sent_num_th": 10, # abandoned
	"grad_sim_th": -0.2,  # abandoned
    "filter_score": 0,
	"filenames":{
        	"dev": "/work/wwang/corpora/CoNLL/original/valid.txt",
        	"test": "/work/wwang/corpora/CoNLL/original/test.txt",
        	"train": "/work/wwang/corpora/CoNLL/original/train.txt"
	},
	"embed_type": "glove",
	"model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_04_28_6f06a57c2f/best-model.pt",
	"embed_folder": "/work/smt2/qfeng/Project/auto_dataset_foundation/embeddings/",
    "ds_rlt_folder": "/work/smt2/qfeng/Project/auto_dataset_foundation/ds_rlt/",
    "sim_folder": "/work/smt2/qfeng/Project/auto_dataset_foundation/sim/",
	"save_embed": False,
	"load_embed": "",
	"load_sent_sim": "",
    "run_emb": False,
    "run_sent_sim": False,
    "run_grad_sim": False,
    "run_resampler": False,
    "Description": "",
    "window":False
}

#taskID = mimesis.Cryptographic().token_hex(5)
#new_dict = dict(base_conf)
#new_dict['taskID'] = taskID
"""
    {"filenames":{
        	"dev": "/work/smt2/qfeng/Project/Data/CoNLL/valid_temp.txt",
        	"test": "/work/smt2/qfeng/Project/Data/CoNLL/test_temp.txt",
        	"train": "/work/smt2/qfeng/Project/Data/CoNLL/train_temp.txt"
	},
    'Description': "test run"}
"""

"""
{"filenames":{
        	"dev": "/work/smt2/qfeng/Project/Data/CoNLL/valid_temp.txt",
        	"test": "/work/smt2/qfeng/Project/Data/CoNLL/test_temp.txt",
        	"train": "/work/smt2/qfeng/Project/Data/CoNLL/train_temp.txt"
	},
	"run_emb": True,
    "run_sent_sim": True,
    "run_grad_sim": True,
    "run_resampler": True,
    "save_embed": False,
    'Description': "test run"}
"""

"""
{
        "save_emb": True,
        "run_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.8,
        "Description": "get sentences similarity sent_th = 0.8"
    },
    {
        "run_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.7,
        "Description": "get sentences similarity sent_th = 0.7"
    },
    {
        "run_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.6,
        "Description": "get sentences similarity sent_th = 0.6"
    },
    {
        "run_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.5,
        "Description": "get sentences similarity sent_th = 0.5"
    }
"""
"""
{"filenames":{
        	"dev": "/work/smt2/qfeng/Project/Data/Ontonotes/ontonotes_dev.conll",
        	"test": "/work/smt2/qfeng/Project/Data/Ontonotes/ontonotes_test.conll",
        	"train": "/work/smt2/qfeng/Project/Data/Ontonotes/ontonotes_train.conll"
	},
	"run_emb": True,
	"dataset": "ontonotes",
    "run_sent_sim": True,
    "save_embed": False,
    "sent_sim_th": 0.8,
    'Description': "Ontonote, get sentences similarity th 0.8"},
"""
"""
{
        "save_embed": True,
        "run_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.8,
	"window": True,
        "Description": "windowed samples, get sample similarity th = 0.8"
    },
"""
to_change = [
{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "3fd769af93_sentsim_glove_sentth0.85_CoNLL_2019_05_19_17_20.pickle",#win 2 sentsim 0.85
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_0ffe7afc5c/best-model.pt", #win 2 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.85,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.85, win size 2, filter score -0.2"
    },
{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "9123062c01_sentsim_glove_sentth0.9_CoNLL_2019_05_19_18_01.pickle",#win 2, sentsim 0.9
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_0ffe7afc5c/best-model.pt", #win 2 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.9,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.8, win size 2, filter score -0.2"
    },
{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "c4d1674044_sentsim_glove_sentth0.95_CoNLL_2019_05_19_17_29.pickle",#win 2, sentsim 0.95
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_0ffe7afc5c/best-model.pt", #win 2 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.95,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.95, win size 2, filter score -0.2"
    },

{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "1dc3526679_sentsim_glove_sentth0.85_CoNLL_2019_05_19_17_09.pickle",#win 1, sentsim 0.85
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_2659b6124f/best-model.pt", #win 1 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.85,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.85, win size 1, filter score -0.2"
    },
{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "6f5d173bf0_sentsim_glove_sentth0.9_CoNLL_2019_05_19_17_17.pickle",#win 1, sentsim 0.9
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_2659b6124f/best-model.pt", #win 1 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.9,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.9, win size 1, filter score -0.2"
    },
{
        "save_embed": False,
        "run_emb": True,
        "run_sent_sim": False,
        "load_sent_sim": "5362563b9e_sentsim_glove_sentth0.95_CoNLL_2019_05_19_17_32.pickle",#win 1, sentsim 0.95
        "model_filename": "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_models/flair04_2019_05_19_2659b6124f/best-model.pt", #win 1 modeel
        "run_grad_sim": True,
        "run_resampler": True,
        "sent_sim_th": 0.95,
	    "window": True,
        "win_size": 2,
        "filter_score": -0.2,
        "Description": "windowed samples, sent sim 0.95, win size 1, filter score -0.2"
    },




]
commands = []
experiments = []
for datasets in to_change:
    new_config = dict(base_conf)
    now = datetime.datetime.now().strftime("%Y_%m_%d")
    taskID = mimesis.Cryptographic().token_hex(5)
    model_tag = "incon_{}_{}".format(now, taskID)
    for k, v in datasets.items():
        new_config[k] = v
    new_config['taskID'] = taskID

    config_name = "/work/smt2/qfeng/Project/auto_dataset_foundation/flair_configs/{}".format(model_tag)
    with open(config_name, "w") as cf:
        json.dump(new_config, cf, indent=4, sort_keys=True)

    job_name = "{}".format(model_tag)
    if new_config["window"]:
        if new_config['embed_type'] == 'bert':
            cluster_command = "echo \"/u/qfeng/Project/auto_dataset_foundation/flair/venv/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor_win.py --config {}\" " \
                              "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l h_vmem=70G -l h_rt=60:00:00 ".format(
                config_name, job_name)
        else:
            cluster_command = "echo \"/u/qfeng/Project/auto_dataset_foundation/flair/venv/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor_win.py --config {}\" " \
                          "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l h_vmem=40G -l h_rt=60:00:00 ".format(
            config_name, job_name)
    else:
        if new_config['embed_type'] == 'bert':
            cluster_command = "echo \"/u/qfeng/Project/auto_dataset_foundation/flair/venv/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor.py --config {}\" " \
                              "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l h_vmem=70G -l h_rt=60:00:00 ".format(
                config_name, job_name)
        else:
            cluster_command = "echo \"/u/qfeng/Project/auto_dataset_foundation/flair/venv/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor.py --config {}\" " \
                          "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l h_vmem=40G -l h_rt=60:00:00 ".format(
            config_name, job_name)

    commands.append(cluster_command)
    experiments.append({"name": taskID, "date": now, "config": model_tag,
                        "job_run": cluster_command,
                        "description": new_config["Description"]
                        })

with open("/work/smt2/qfeng/Project/auto_dataset_foundation/run/run_experiments_{}.sh".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
          "w") as exp:
    for c in commands:
        exp.write(c)
        exp.write("\n")

result = pd.DataFrame(experiments)
result.to_csv("/work/smt2/qfeng/Project/auto_dataset_foundation/logs/experiments/experiments_incon_{}.csv".format(now), mode="a")

"""
cluster_command = "echo \"/u/qfeng/Project/work_37/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor.py --config {}\" " \
                          "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l gpu=1 -l h_vmem=70G -l qname=*1080* -l h_rt=60:00:00 ".format(
            config_name, job_name)
"""
