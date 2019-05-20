from typing import Dict, List
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus, Sentence, Label, Token
import torch
import pickle
import datetime

import bob

from torch.nn.modules.distance import CosineSimilarity
from flair.embeddings import BertEmbeddings, WordEmbeddings

class Sent_Cluster:
    def __init__(self,params):
        self.sent_dict: Dict[int, Sentence] = {}
        self.id_sample_2_sent: Dict[int, int] = {}
        self.sample_dict: Dict[int, Sentence] = {}
        self.sample_embd: Dict[int, torch.Tensor] = {}
        self.params = params
        if torch.cuda.is_available():
           self.device = torch.device('cuda:0')
        else:
           self.device = torch.device('cpu')
        pass

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
                self.sample_embd[id] = sent_vec
        elif self.params['embed_type'] == 'bert':
            bert_embd = BertEmbeddings('bert-base-cased')
            for id in range(len(self.sample_dict)):
                sent = self.sample_dict[id]
                bert_embd.embed(sent)
                sent_vec = torch.zeros(self.max_len * bert_embd.embedding_length, device=self.device)
                sent_vec[:len(sent) * bert_embd.embedding_length] = torch.cat(
                    [token.get_embedding().unsqueeze(0) for token in sent], 0).view(-1)
                del sent_vec
                self.sample_embd[id] = sent_vec
        else:
            print('unexpected embedding type!')
            return
        if self.params['save_embed']:
            with open(self.params['embed_folder'] + '{}_{}_{}_win.pickle'.format(self.params['taskID'],
                                                                             self.params['embed_type'],
                                                                             datetime.datetime.now().strftime(
                                                                                     "%Y_%m_%d_%H_%M")),
                      'wb') as handle:
                pickle.dump(self.sample_embd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("finish embedding: {}".format(datetime.datetime.now().strftime("%d:%H:%M")))
        return

    def kmean_cluster(self):
        #TODO: implement kmean_cluster according to param['cluster_comp']
        # mixture_mean{mixture_id: mean of mixture}
        pass

    def gmm_cluster(self):
        # mixture_mean{mixture_id: mean of mixture}
        pass

    def save_cluster(self):
        pass

    def load_cluster(self):
        pass


    def visualize(self):
        """
        1.distribution of majority ratio of components
        2.distribution of means (2-Norm)
        """
        pass
