import json
import datetime
import mimesis
import pandas as pd

base_conf = {
    "dataset": 'CoNLL',
	"sent_sim_th": 0.1,
	"sent_num_th": 10,
	"grad_sim_th": -0.2,  # abandoned
    'filter_score': 0,
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
    "Description": ""
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
{
        "load_emb": "c105dcd5a1_glove_2019_04_29_14_14.pickle",
        "run_sent_sim": True,
        "sent_sim_th": 0.8,
        'Description:': "get sentences similarity sent_th = 0.8"
    },
    {
        "load_emb": "c105dcd5a1_glove_2019_04_29_14_14.pickle",
        "run_sent_sim": True,
        "sent_sim_th": 0.7,
        'Description:': "get sentences similarity sent_th = 0.7"
    },
    {
        "load_emb": "c105dcd5a1_glove_2019_04_29_14_14.pickle",
        "run_sent_sim": True,
        "sent_sim_th": 0.6,
        'Description:': "get sentences similarity sent_th = 0.6"
    },
    {
        "load_emb": "c105dcd5a1_glove_2019_04_29_14_14.pickle",
        "run_sent_sim": True,
        "sent_sim_th": 0.5,
        'Description:': "get sentences similarity sent_th = 0.5"
    }
"""
"""
{"filenames":{
        	"dev": "/work/smt2/qfeng/Project/Data/CoNLL/valid_temp.txt",
        	"test": "/work/smt2/qfeng/Project/Data/CoNLL/test_temp.txt",
        	"train": "/work/smt2/qfeng/Project/Data/CoNLL/train_temp.txt"
	},
    "run_sent_sim": True,
    "run_grad_sim": True,
    "run_resampler": True,
    "save_embed": False,
    'Description': "test run"}
"""

to_change = [

    #{"sent_sim_th": 0.8,'Description:': "get sentences similarity sent_th = 0.8"},
    #{"sent_sim_th": 0.7,'Description:': "get sentences similarity sent_th = 0.7"},
    #{"sent_sim_th": 0.6,'Description:': "get sentences similarity sent_th = 0.6"},
    #{"sent_sim_th": 0.5,'Description:': "get sentences similarity sent_th = 0.5"}
    #{"embed_type": 'bert', "Description": "get bert embedding"}

    {
        "save_emb": True,
        "run_sent_sim": True,
        "sent_sim_th": 0.8,
        "Description": "get sentences similarity sent_th = 0.8"
    },
    {
        "run_sent_sim": True,
        "sent_sim_th": 0.7,
        "Description": "get sentences similarity sent_th = 0.7"
    },
    {
        "run_sent_sim": True,
        "sent_sim_th": 0.6,
        "Description": "get sentences similarity sent_th = 0.6"
    },
    {
        "run_sent_sim": True,
        "sent_sim_th": 0.5,
        "Description": "get sentences similarity sent_th = 0.5"
    }

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
    if new_config['embed_type'] == 'bert':
        cluster_command = "echo \"/u/qfeng/Project/work_37/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor.py --config {}\" " \
                          "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l gpu=1 -l h_vmem=70G -l qname=*1080* -l h_rt=60:00:00 ".format(
            config_name, job_name)
    else:
        cluster_command = "echo \"/u/qfeng/Project/work_37/bin/python3 /u/qfeng/Project/auto_dataset_foundation/flair/flair/Inconsist_Ana/inconsist_analyzor.py --config {}\" " \
                      "| qsub -o /work/smt2/qfeng/Project/auto_dataset_foundation/logs/flair -e /work/smt2/qfeng/Project/auto_dataset_foundation/logs/errors -N {} -l gpu=1 -l h_vmem=40G -l qname=*1080* -l h_rt=60:00:00 ".format(
        config_name, job_name)

    commands.append(cluster_command)
    experiments.append({"name": taskID, "date": now, "config": config_name,
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

