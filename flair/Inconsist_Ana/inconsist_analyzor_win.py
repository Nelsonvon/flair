import os, sys
sys.path.append("/u/qfeng/Project/auto_dataset_foundation/flair/")
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus, Sentence, Label, Token
from flair.models import SequenceTagger
from typing import Dict, List
import torch
from torch.nn.modules.distance import CosineSimilarity
from flair.embeddings import BertEmbeddings, WordEmbeddings
import argparse
import re
import json
import pickle
import datetime
import multiprocessing
import re
import mimesis
from flair import device

"""
27.04:
Reduce repeated calculation
cal_sent_sim go back to uni-prosess
improve naming of return
"""
"""
30.04:
Drop test/dev set from filtering
"""
"""
09.05:
switch to cpu computation because of bugs when using CUDNN.
convert to variant processing windowed samples in stead of full sentences.
"""

def read_conll_format(filenames):
    sentences = []
    tags = []
    max_len = 0
    for f in filenames:
        filename = filenames[f]
        with open(filename) as fn:
            s, t = [], []
            for line in fn:
                line = line.strip()
                if line == "":
                    if len(s) > max_len:
                        max_len = len(s)
                    sentences.append(Sentence(" ".join(s)))
                    tags.append(t)
                    s, t = [], []
                    continue
                fields = re.split("\s+", line)

                s.append(fields[0])
                t.append(fields[1])

    print(len(sentences))
    return sentences, tags, max_len


def get_grad(model: SequenceTagger, sentences: List[Sentence], grad_layer: str, sort=True):
    model.zero_grad()

    model.embeddings.embed(sentences)

    # if sorting is enabled, sort sentences by number of tokens
    if sort:
        sentences.sort(key=lambda x: len(x), reverse=True)

    lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
    tag_list: List = []
    longest_token_sequence_in_batch: int = lengths[0]

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # initialize zero-padded word embeddings tensor
    sentence_tensor = torch.zeros([len(sentences),
                                   longest_token_sequence_in_batch,
                                   model.embeddings.embedding_length],
                                  dtype=torch.float, device=device)
    """
    I don't know what happened that
    """
    for s_id, sentence in enumerate(sentences):
        # fill values with word embeddings
        sentence_tensor[s_id][:len(sentence)] = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence], 0)

        # get the tags in this sentence
        tag_idx: List[int] = [model.tag_dictionary.get_idx_for_item(token.get_tag(model.tag_type).value)
                              for token in sentence]
        # add tags as tensor
        tag = torch.LongTensor(tag_idx).to(device)
        tag_list.append(tag)

    sentence_tensor = sentence_tensor.transpose_(0, 1)

    # --------------------------------------------------------------------
    # FF PART
    # --------------------------------------------------------------------
    if model.use_dropout > 0.0:
        sentence_tensor = model.dropout(sentence_tensor)
    if model.use_word_dropout > 0.0:
        sentence_tensor = model.word_dropout(sentence_tensor)
    if model.use_locked_dropout > 0.0:
        sentence_tensor = model.locked_dropout(sentence_tensor)

    if model.relearn_embeddings:
        sentence_tensor = model.embedding2nn(sentence_tensor)

    if model.use_rnn:
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)

        rnn_output, hidden = model.rnn(packed)

        # crf_tensor = torch.zeros(model.embeddings.embedding_length,requires_grad=True)
        # crf_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output)

        if model.use_dropout > 0.0:
            sentence_tensor = model.dropout(sentence_tensor)
        # word dropout only before LSTM - TODO: more experimentation needed
        # if model.use_word_dropout > 0.0:
        #     sentence_tensor = model.word_dropout(sentence_tensor)
        if model.use_locked_dropout > 0.0:
            sentence_tensor = model.locked_dropout(sentence_tensor)
        sentence_tensor.retain_grad()
    # sentence_tensor.retain_grad()
    features = model.linear(sentence_tensor)
    # features.retain_grad()

    loss = model._calculate_loss(features.transpose_(0, 1), lengths, tag_list)
    loss.backward()

    return [p.grad for p in list(model.linear.parameters())]
    # return features.grad.data
    # return  sentence_tensor.grad.data


class Inconsist_Analyzor:
    def __init__(self, params):
        # self.corpus : TaggedCorpus
        self.model: SequenceTagger
        self.sent_dict: Dict[int,Sentence] = {}
        self.id_sample_2_sent: Dict[int,int] = {}
        self.sample_dict: Dict[int, Sentence] = {}
        self.sent_embd: Dict[int, torch.Tensor] = {}
        self.sample_sim: Dict[int, Dict[int, float]] = {}
        self.grad_sim: Dict[int, Dict[int, float]] = {}
        self.params = params

        self.device = torch.device('cpu')
        # if torch.cuda.is_available():
        #    self.device = torch.device('cuda:0')
        # else:
        #    self.device = torch.device('cpu')
        self.cs: CosineSimilarity = CosineSimilarity(dim=1, eps=1e-6)
        return

    def get_corpus(self):
        self.corpus = NLPTaskDataFetcher.load_corpus(task=NLPTask['TAC'], files=self.params['filenames'])
        self.sent_dict = {sent_id: sent for (sent_id, sent) in
                          enumerate(self.corpus.get_all_sentences()[:len(self.corpus.train)])}
        if 'win_size' in self.params:
            win_size = self.params['win_size']
        else:
            win_size = 3
        count = 0
        for id in self.sent_dict:
            #extract windowed samples from sentence
            sent = self.sent_dict[id]
            span_start_pos = 0
            span_length = 0
            previous_tag_value: str = 'O'
            # in_span: bool =False
            current_loc = -1
            for token in sent.tokens:
                current_loc += 1
                tag: Label = token.get_tag('ner')
                tag_value = tag.value

                # non-set tags are OUT tags
                if tag_value == '' or tag_value == 'O':
                    tag_value = 'O-'

                # anything that is not a BIOES tag is a SINGLE tag
                if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-']:
                    tag_value = 'S-' + tag_value

                # anything that is not OUT is IN
                in_span = False
                if tag_value[0:2] not in ['O-']:
                    in_span = True
                    # span_length +=1

                # single and begin tags start a new span
                starts_new_span = False
                if tag_value[0:2] in ['B-', 'S-']:
                    starts_new_span = True

                if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
                    starts_new_span = True

                if (not in_span or starts_new_span) and (span_length != 0):
                    sentence: Sentence = Sentence()
                    # not in a span but the buffer is not empty: define a windowed sample for that
                    for pos in range(span_start_pos - win_size, span_start_pos + span_length + win_size):
                        if (pos < 0) or (pos >= len(sent)):
                            text = '##'
                            tag = 'O'
                        else:
                            text = sent.tokens[pos].text
                            if (pos < span_start_pos) or (pos > span_start_pos + span_length):
                                tag = 'O'
                            else:
                                tag = sent.tokens[pos].get_tag('ner').value
                        t = Token(text)
                        t.add_tag('ner', tag)
                        sentence.add_token(t)
                    sentence.infer_space_after()
                    self.sample_dict[count] = sentence
                    self.id_sample_2_sent[count] = id
                    count +=1
                    # print("print win:[{}-{}]".format(str(span_start_pos-win_size),str(span_start_pos+span_length+win_size)))
                    span_length = 0
                if in_span:
                    span_length += 1
                if starts_new_span:
                    span_start_pos = current_loc
                previous_tag_value = tag_value
                # print("current_pos:{}, start_pos: {}, length:{}".format(str(current_loc),str(span_start_pos),str(span_length)))

            if (not in_span or starts_new_span) and (span_length != 0):
                sentence: Sentence = Sentence()
                # not in a span but the buffer is not empty: define a windowed sample for that
                for pos in range(span_start_pos - win_size, span_start_pos + span_length + win_size):
                    if (pos < 0) or (pos >= len(sent)):
                        text = '##'
                        tag = 'O'
                    else:
                        text = sent.tokens[pos].text
                        if (pos < span_start_pos) or (pos > span_start_pos + span_length):
                            tag = 'O'
                        else:
                            tag = sent.tokens[pos].get_tag('ner').value
                    t = Token(text)
                    t.add_tag('ner', tag)
                    sentence.add_token(t)
                sentence.infer_space_after()
                self.sample_dict[count] = sentence
                self.id_sample_2_sent[count] = id
                count += 1
                # print("print win:[{}-{}]".format(str(span_start_pos-win_size),str(span_start_pos+span_length+win_size)))
                span_length = 0

        print("sentences in corpus: {}".format(str(len(self.sent_dict))))
        print("windowed samples in corpus: {}".format(str(len(self.sample_dict))))
        print("finished read corpus and window extraction ")
        return

    def get_model(self):
        self.model = SequenceTagger.load_from_file(self.params['model_filename'])
        pass

    def find_sim_sents(self, id):
        sent_sim_id = {}
        for new_id in range(len(self.sample_dict)):
            if id != new_id:
                # print("id= {}, new_id={}".format(str(id),str(new_id)))
                dist = float(self.cs.forward(self.sent_embd[id].unsqueeze(0), self.sent_embd[new_id].unsqueeze(0)))
                if dist >= self.params['sent_sim_th']:
                    sent_sim_id[new_id] = dist
        if id % 500 == 0:
            print("finished Epoch {}: Time: {}".format(str(id), datetime.datetime.now().strftime(
                "%H:%M")))
            if len(sent_sim_id) > 0:
                print("test score: {}".format(str(sent_sim_id[list(sent_sim_id.keys())[0]])))
        return id, sent_sim_id

    def cal_sent_sim(self):
        # similarity can base on either flair embedding or bert embedding
        # pick the most similar sents for each sent.

        for id in range(len(self.sample_dict)):
            for new_id in range(id, len(self.sample_dict)):
                if id not in self.sample_sim:
                    self.sample_sim[id] = {}
                if (id != new_id) and (new_id not in self.sample_sim[id]):
                    dist = self.cs.forward(self.sent_embd[id].unsqueeze(0), self.sent_embd[new_id].unsqueeze(0))
                    if dist >= self.params['sent_sim_th']:
                        if id not in self.sample_sim:
                            self.sample_sim[id] = {}
                        self.sample_sim[id][new_id] = dist
                        if new_id not in self.sample_sim:
                            self.sample_sim[new_id] = {}
                        self.sample_sim[new_id][id] = dist
            if id % 500 == 0:
                print("finished Epoch {}: Time: {}".format(str(id), datetime.datetime.now().strftime(
                    "%H:%M")))
                # print("\nsent1: {}\nsent2: {}\n Similarity: {}".format(self.sample_dict[id].to_plain_string(),self.sample_dict[new_id].to_plain_string(), dist))
            #print(self.sample_sim)
        with open(
                self.params['sim_folder'] + 'sent_sim/{}_sentsim_{}_sentth{}_{}_{}.pickle'.format(self.params['taskID'],
                                                                                                  self.params[
                                                                                                      'embed_type'],
                                                                                                  str(self.params[
                                                                                                          'sent_sim_th']),
                                                                                                  self.params[
                                                                                                      'dataset'],
                                                                                                  datetime.datetime.now().strftime(
                                                                                                      "%Y_%m_%d_%H_%M")),
                'wb') as handle:
            pickle.dump(self.sample_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("finish sent_sim: {}".format(datetime.datetime.now().strftime("%d_%H_%M")))
        return

    def cal_sent_sim_multips(self):
        # similarity can base on either flair embedding or bert embedding
        # pick the most similar sents for each sent.

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        tasks = [id for id in range(len(self.sample_dict))]
        result = []
        # for id in range(len(self.sample_dict)):
        #   result.append(pool.apply_async(self.find_sim_sents, args=(id,)))
        # self.find_sim_sents(id)
        result = pool.map(self.find_sim_sents, tasks)

        # print("\nsent1: {}\nsent2: {}\n Similarity: {}".format(self.sample_dict[id].to_plain_string(),self.sample_dict[new_id].to_plain_string(), dist))
        # print(self.sample_sim)
        print("subprocessing may not finish yet")
        pool.close()
        pool.join()
        for i in result:
            id, sent_sim_id = i
            # print(id)
            self.sample_sim[id] = sent_sim_id
        print("multiprocessing finished")
        # print(self.sample_sim)
        with open(
                self.params['sim_folder'] + 'sent_sim/{}_sentsim_{}_sentth{}_{}_{}_win.pickle'.format(self.params['taskID'],
                                                                                                  self.params[
                                                                                                      'embed_type'],
                                                                                                  str(self.params[
                                                                                                          'sent_sim_th']),
                                                                                                  self.params[
                                                                                                      'dataset'],
                                                                                                  datetime.datetime.now().strftime(
                                                                                                      "%Y_%m_%d_%H_%M")),
                'wb') as handle:
            pickle.dump(self.sample_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def cal_grad_sim(self) -> int:  # return number of inconsistent pairs
        grad_dict: Dict[int, torch.Tensor] = {}
        count = 0
        for id in self.sample_sim:
            if id not in grad_dict:
                sent = self.sample_dict[id]
                grad: List[torch.Tensor] = get_grad(self.model, sentences=[sent], grad_layer='output')
                grad = torch.cat([grad[0].view(1, -1), grad[1].view(1, -1)], 1)
                # grad: torch.Tensor = self.model.get_grad(sentences=[sent], grad_layer='output')
                # print(grad.size())
                # grad = grad.view(1,-1)
                grad_dict[id] = grad
            min_grad_sim = 1
            if id not in self.grad_sim:
                self.grad_sim[id] = {}
            for new_id in self.sample_sim[id]:
                if (new_id != id) and (new_id not in self.grad_sim[id]) and (
                        self.sample_sim[id][new_id] >= self.params['sent_sim_th']):
                    # print("find suitable pair")
                    if new_id not in grad_dict:
                        sent = self.sample_dict[new_id]
                        self.model.train()
                        grad: List[torch.Tensor] = self.model.get_grad(sentences=[sent], grad_layer='output')
                        grad = torch.cat([grad[0].view(1, -1), grad[1].view(1, -1)], 1)
                        # grad: torch.Tensor = self.model.get_grad(sentences=[sent], grad_layer='output')
                        # grad = grad.view(1, -1)
                        grad_dict[new_id] = grad
                    dist = self.cs.forward(grad_dict[id], grad_dict[new_id])
                    if dist < min_grad_sim:
                        min_grad_sim = dist
                    if dist < self.params['grad_sim_th']:
                        count += 1
                        #print("\nsent1: {}\nsent2: {}\n Similarity: {}".format(self.sample_dict[id].to_plain_string()+str(self.sample_dict[id].get_spans('ner')),
                        #                                                      self.sample_dict[new_id].to_plain_string()+str(self.sample_dict[new_id].get_spans('ner')), dist))
                    if id not in self.grad_sim:
                        self.grad_sim[id] = {}
                    self.grad_sim[id][new_id] = dist
                    if new_id not in self.grad_sim:
                        self.grad_sim[new_id] = {}
                    self.grad_sim[new_id][id] = dist
            # if len(self.sample_sim[id])>0:
            #    print("End processing id: {}, min grad sim: {}".format(str(id),str(min_grad_sim)))
        with open(
                self.params['sim_folder'] + 'grad_sim/{}_gradsim_{}_sentth{}_{}_{}_win.pickle'.format(self.params['taskID'],
                                                                                                  self.params[
                                                                                                      'embed_type'],

                                                                                                  str(self.params[
                                                                                                          'sent_sim_th']),
                                                                                                  self.params[
                                                                                                      'dataset'],
                                                                                                  datetime.datetime.now().strftime(
                                                                                                      "%Y_%m_%d_%H_%M")),
                'wb') as handle:
            pickle.dump(self.grad_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("finish grad_sim: {}".format(datetime.datetime.now().strftime("%d_%H_%M")))
        return count

    def resampler(self):
        # TODO: modification for window_level
        # TODO: limit the number of similar sentences
        """
        delete sentences in train and dev set which cause inconsistency and output the downsampled datasets
        :return: use a dict to collect the statistics and return
        """
        stat: Dict = {}
        self.incon_id: Dict[str, List] = {}

        def is_inconsistent(sent_sim: Dict[int, float], grad_sim: Dict[int, float], th: float) -> bool:
            # TODO: check whether the id of two dict are indentical
            score = 0
            for id in sent_sim:
                score += sent_sim[id] * grad_sim[id]
            score = score / len(sent_sim)
            if (score <= th):
                return True
            else:
                return False

        """
        self.incon_id['train'] = []
        for id in self.train_id:
            if(id in self.sample_sim) and (id in self.grad_sim):
                if(is_inconsistent(self.sample_sim[id],self.grad_sim[id], self.params['filter_score'])):
                    self.incon_id['train'].append(id)

        self.incon_id['dev'] = []
        for id in self.dev_id:
            if (id in self.sample_sim) and (id in self.grad_sim):
                if (is_inconsistent(self.sample_sim[id], self.grad_sim[id], self.params['filter_score'])):
                    self.incon_id['dev'].append(id)

        stat['del_sents'] = {'train': len(self.incon_id['train']), 'dev': len(self.incon_id['dev'])}
        """
        self.incon_id['train'] = []
        for id in self.sample_dict:
            if (id in self.sample_sim) and (id in self.grad_sim):
                if (is_inconsistent(self.sample_sim[id], self.grad_sim[id], self.params['filter_score'])):
                    self.incon_id['train'].append(self.id_sample_2_sent[id])
        stat['del_sents'] = {'train': len(self.incon_id['train'])}
        print("finish resampler: {}, deleted {} sentences".format(datetime.datetime.now().strftime("%d_%H_%M"),
                                                                  str(len(self.incon_id['train']))))
        return stat

    def visualize_sent_grad(self):
        pass

    def cal_embedding(self):
        #
        lengths: List[int] = [len(sentence.tokens) for sentence in self.sample_dict.values()]
        self.max_len = max(lengths)
        # longest_token_sequence_in_batch: int = lengths[0]
        if self.params['embed_type'] == 'glove':
            glove_embd = WordEmbeddings('glove')
            for id in range(len(self.sample_dict)):
                sent = self.sample_dict[id]
                glove_embd.embed(sent)
                sent_vec = torch.zeros(self.max_len * glove_embd.embedding_length, device=self.device)
                sent_vec[:len(sent) * glove_embd.embedding_length] = torch.cat(
                    [token.get_embedding().unsqueeze(0) for token in sent], 0).view(-1)
                self.sent_embd[id] = sent_vec
        elif self.params['embed_type'] == 'bert':
            bert_embd = BertEmbeddings('bert-base-cased')
            for id in range(len(self.sample_dict)):
                sent = self.sample_dict[id]
                bert_embd.embed(sent)
                sent_vec = torch.zeros(self.max_len * bert_embd.embedding_length, device=self.device)
                sent_vec[:len(sent) * bert_embd.embedding_length] = torch.cat(
                    [token.get_embedding().unsqueeze(0) for token in sent], 0).view(-1)
                del sent_vec
                self.sent_embd[id] = sent_vec
        else:
            print('unexpected embedding type!')
            return
        if self.params['save_embed']:
            with open(self.params['embed_folder'] + '{}_{}_{}_win.pickle'.format(self.params['taskID'],
                                                                             self.params['embed_type'],
                                                                             datetime.datetime.now().strftime(
                                                                                     "%Y_%m_%d_%H_%M")),
                      'wb') as handle:
                pickle.dump(self.sent_embd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("finish embedding: {}".format(datetime.datetime.now().strftime("%d:%H:%M")))
        return

    def load_pickles(self):
        if self.params['load_embed'] != "":
            with open(self.params['embed_folder'] + self.params['load_embed'], 'rb') as handle:
                self.sent_embd = pickle.load(handle)
        if self.params['load_sent_sim'] != "":
            with open(self.params['sim_folder'] + 'sent_sim/' + self.params['load_sent_sim'], 'rb') as handle:
                self.sample_sim = pickle.load(handle)
        return

    def save_downsample(self):
        # TODO: output the downsampled sets
        # save training set
        with open(self.params['ds_rlt_folder'] + '{}_train_sentth{}_score{}_win.txt'.format(self.params['taskID'],
                                                                                        self.params['sent_sim_th'],
                                                                                        self.params['filter_score']),
                  'w') as fout:
            for id in self.sample_dict:
                if id not in self.incon_id['train']:
                    sent = self.sample_dict[id]
                    for token in sent.tokens:
                        # print(token.text)
                        # print(token.get_tag('ner').value)
                        w = token.text
                        t = token.get_tag('ner').value
                        if re.match("S-.*", t):
                            t = re.sub("S-", "B-", t)
                        elif re.match("E-.*", t):
                            t = re.sub("E-", "I-", t)
                        fout.write("{}\t{}\n".format(w, t))
                    fout.write('\n')
        """
        with open(self.params['ds_rlt_folder']+'{}_dev.txt'.format(self.params['taskID']),'wb') as fout:
            for id in self.dev_id:
                if id not in self.incon_id['dev']:
                    sent = self.sample_dict[id]
                    for token_id in range(len(sent)):
                        token = sent.get_token(token_id)
                        fout.write('{}\t{}\n'.format(token.text,token.get_tag('ner')))
                fout.write('\n')
        """
        pass

    def process(self):
        self.get_corpus()

        self.load_pickles()
        if self.params['load_embed'] == "" and self.params['run_emb']:
            self.cal_embedding()
        if self.params['load_sent_sim'] == "" and self.params['run_sent_sim']:
            self.cal_sent_sim()
        if self.params['run_grad_sim']:
            self.get_corpus()
            self.get_model()
            self.cal_grad_sim()
        if self.params['run_resampler']:
            self.resampler()
            self.save_downsample()
        return

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", default="/home/nelson/Data/auto_database_foundation/flair_configs/incon_2019_05_06_b0da16e315")
    args = arg_parser.parse_args()
    with open(args.config) as cfg:
        params = json.load(cfg)
    print(params)
    # params['taskID'] = taskID
    # params['dataset'] = 'CoNLL'
    analyst = Inconsist_Analyzor(params)
    analyst.process()

# else:
#    arg_parser = argparse.ArgumentParser()
#    arg_parser.add_argument("--config", default="/home/nelson/Data/auto_database_foundation/flair_configs/2019_04_22")
#    args = arg_parser.parse_args()
#    with open(args.config) as cfg:
#        params = json.load(cfg)
#    analyst = Inconsist_Analyzor(params)
#    analyst.get_corpus()
#    analyst.cal_embedding()
#    analyst.cal_sent_sim()
