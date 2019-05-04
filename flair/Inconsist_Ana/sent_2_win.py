import os, sys
sys.path.append("/u/qfeng/Project/auto_dataset_foundation/flair/")
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from typing import Dict,List
from flair.data import Sentence, Token, Label
import re

"""
Convert datasets with sentences to with windows
Used to train the ner model for windows

label only one words in a sentence and include padding  
"""


def get_sents(filename)->List[Sentence]:
    col_format = columns = {0: 'text', 1: 'ner'} # TODO: formats for other corpura
    return NLPTaskDataFetcher.read_column_data(filename, col_format)

def save_wins(sents: List[Sentence], filename:str, win_size):
    with open(filename,'w+') as fout:
        for sent in sents:
            #print(sent.to_plain_string())
            span_start_pos = 0
            span_length = 0
            previous_tag_value: str = 'O'
            #in_span: bool =False
            current_loc = -1
            for token in sent.tokens:
                current_loc +=1
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
                    #span_length +=1

                # single and begin tags start a new span
                starts_new_span = False
                if tag_value[0:2] in ['B-', 'S-']:
                    starts_new_span = True

                if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
                    starts_new_span = True

                if (not in_span or starts_new_span)and (span_length!=0):
                    #not in a span but the buffer is not empty: just went through a span and should output it.
                    for pos in range(span_start_pos-win_size,span_start_pos+span_length+win_size):
                        if (pos<span_start_pos) or (pos>=len(sent)):
                            text = '##'
                            tag = 'O'
                        else:
                            text = sent.tokens[pos].text
                            if (pos<span_start_pos) or (pos>span_start_pos+span_length):
                                tag = 'O'
                            else:
                                tag = sent.tokens[pos].get_tag('ner').value
                            if re.match("S-.*", tag):
                                tag = re.sub("S-", "B-", tag)
                            elif re.match("E-.*", tag):
                                tag = re.sub("E-", "I-", tag)
                        fout.write("{}\t{}\n".format(text, tag))
                    fout.write(('\n'))
                    #print("print win:[{}-{}]".format(str(span_start_pos-win_size),str(span_start_pos+span_length+win_size)))
                    span_length = 0
                if in_span:
                    span_length +=1
                if starts_new_span:
                    span_start_pos = current_loc
                previous_tag_value = tag_value
                #print("current_pos:{}, start_pos: {}, length:{}".format(str(current_loc),str(span_start_pos),str(span_length)))

            if (not in_span or starts_new_span) and (span_length!=0):
                # not in a span but the buffer is not empty: just went through a span and should output it.
                for pos in range(span_start_pos - win_size, span_start_pos + span_length + win_size+1):
                    if (pos < 0) or (pos >= len(sent)):
                        text = '##'
                        tag = 'O'
                    else:
                        text = sent.tokens[pos].text
                        if (pos < span_start_pos) or (pos > span_start_pos + span_length):
                            tag = 'O'
                        else:
                            tag = sent.tokens[pos].get_tag('ner').value
                        if re.match("S-.*", tag):
                            tag = re.sub("S-", "B-", tag)
                        elif re.match("E-.*", tag):
                            tag = re.sub("E-", "I-", tag)
                    fout.write("{}\t{}\n".format(text, tag))
                fout.write(('\n'))
                #print("print win:[{}-{}]".format(str(span_start_pos - win_size),
                                                 #str(span_start_pos + span_length + win_size + 1)))
    return

sents_temp = get_sents('/home/nelson/Data/auto_database_foundation/datasets/CoNLL/train.txt')
save_wins(sents_temp,'/home/nelson/Data/auto_database_foundation/datasets/CoNLL/train_win.txt', 3)
sents_temp = get_sents('/home/nelson/Data/auto_database_foundation/datasets/CoNLL/valid.txt')
save_wins(sents_temp,'/home/nelson/Data/auto_database_foundation/datasets/CoNLL/valid_win.txt', 3)
sents_temp = get_sents('/home/nelson/Data/auto_database_foundation/datasets/CoNLL/test.txt')
save_wins(sents_temp,'/home/nelson/Data/auto_database_foundation/datasets/CoNLL/test_win.txt', 3)
