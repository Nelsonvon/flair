from typing import List, Dict
import csv
import argparse
import re
import datetime
import pickle

from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, PooledFlairEmbeddings, CharacterEmbeddings, \
    BertEmbeddings

data_folder = "/home/nelson/Data/auto_database_foundation/datasets/"
filenames_conll= {
        "dev": data_folder+"CoNLL/valid.txt",
        "test": data_folder+"CoNLL/test.txt",
        "train": [
            data_folder+"CoNLL/train.txt"
            #"/work/wwang/corpora/CoNLL/original/train.txt",
            #"/work/wwang/corpora/OntoNotes/training/ontonotes_full_scored_by_forward_lm.tagbymodel.scorebased.removeO.conll"
        ]
}

filenames_ontonotes= {
        "dev": data_folder+"Ontonotes/ontonotes_dev.conll",
        "test": data_folder+"Ontonotes/ontonotes_test.conll",
        "train": [
            data_folder+"Ontonotes/ontonotes_train.conll"
            #"/work/wwang/corpora/CoNLL/original/train.txt",
            #"/work/wwang/corpora/OntoNotes/training/ontonotes_full_scored_by_forward_lm.tagbymodel.scorebased.removeO.conll"
        ]
}

#class Token_desc:
def decoder(encoded_name: str, kb_type:str, comp_type: str) -> str:
    if kb_type is 'Simple' or kb_type is 'Types':
        if comp_type is 's' : # subject decoder:
            name = encoded_name.replace('<','')
            name = name.replace('>','')
            name = name.replace('_',' ')
            return  name
        if comp_type is 'p':
            return encoded_name
        if comp_type is 'o' : # object decoder:
            name = encoded_name.replace('<','')
            name = name.replace('>','')
            """
            yatoSimpleTypes' object start with wikicat_ or wikinet_, if not, report new type
            """
            if name.startswith('wikicat_') :
                name = name[len('wikicat_'):]
            elif name.startswith('wordnet_'): # wordnet format: <wordnet_person_100007846>
                name = name[len('wordnet_'):]
                name = name[:-len('_100007846')]
            #else: only yagoGeoEntity
                #print('############### Found new object type: {}\n'.format(name))
            name = name.replace('_',' ')
            return  name
    elif kb_type is 'Facts':
        if comp_type is 's' : # subject decoder:
            name = encoded_name.replace('<','')
            name = name.replace('>','')
            name = name.replace('_',' ')
            return  name
        if comp_type is 'p':
            name = encoded_name.replace('<', '')
            name = name.replace('>', '')
            name = name.replace('_', ' ')
            return name
        if comp_type is 'o' : # object decoder:
            name = encoded_name.replace('<', '')
            name = name.replace('>', '')
            name = name.replace('_', ' ')
            return  name
    elif kb_type is 'Date':
        """
        format: <1st_Light_Car_Patrol_(Australia)>	<wasDestroyedOnDate>	"1919-##-##"^^xsd:date	1919.0000
        """
        if comp_type is 's' : # subject decoder:
            name = encoded_name.replace('<','')
            name = name.replace('>','')
            name = name.replace('_',' ')
            return  name
        if comp_type is 'p':
            name = encoded_name.replace('<', '')
            name = name.replace('>', '')
            name = name.replace('_', ' ')
            return name
        if comp_type is 'o' : # object decoder:
            return  encoded_name[1:-11]

def isNeedSubject(subject: str) -> bool:
    # delete subjects from foreign language
    if subject.startswith('/',2): # format: two characters abbr./subject
        return False
    elif '/' in subject:
        #print('$$$$$$$$$$$$$ Exception of subject filtering: {}\n'.format(subject))
        return True
    else:
        return  True

class YAGO_data:

    def __init__(self, params,kb_type):
        #format:
        self.kb : Dict[str,List[str]] = {}
        self.kb_folder = params.folder
        self.kb_type = kb_type
        pass

    def load_kb_tsv(self, kb_name) -> Dict:
        """
        Format of different files:
        - yagoFacts:
                <id_J4g7!GNccC_5yH_seYbWPlbge>  <Network_Rail>  <owns>  <Headstone_Lane_railway_station>
        - yagoDateFacts:
                <id_55R1FAawhE_8VX_N3WATqufJe>  <Wedgewood_Village_Amusement_Park>      <wasDestroyedOnDate>    "1969-##-##"^^xsd:date  1969.0000
        - yagoTypes:
                <id_wGHfubCwBs_KCM_wTujFTpmfI>  <Jean-Baptiste-Joseph_Gobel>    rdf:type        <wikicat_Roman_Catholic_archbishops_in_France>
        - yagoTypes:
                        <es/Alberto_Ruiz_Largo> rdf:type        <wikicat_Sporting_de_Gijón_B_players>
        - yagoTransitiveType:
                <id_QCt3Vm7wgc_KCM_D8FCd!kRW4>  <Saccobolus_glaber>     rdf:type        <wikicat_Fungi>

        only english mentions would be used so far (no data in form de/... or fr/...)
        Upper cases & lower cases transformation
        """
        #if kb_name == 'yagoSimpleTypes.tsv':
        with open(self.kb_folder+kb_name) as file:
            file.readline()
            for line in file:
                if self.kb_type == 'Simple':
                    shift_s = 0
                    shift_p = 0
                    shift_o = 0
                else:
                    shift_s = 1
                    shift_p = 1
                    shift_o = 1
                content = line.split()
                f_subject = decoder(content[0+shift_s],self.kb_type,'s')
                f_predicate = decoder(content[1+shift_p],self.kb_type,'p')
                try:
                    f_object = decoder(content[2+shift_o],self.kb_type,'o')
                except:
                    print(content)
                if isNeedSubject(f_subject):
                    if f_subject in self.kb:
                        self.kb[f_subject].append(' '.join([f_predicate,f_object]))
                    else:
                        #print(f_predicate)
                        #print(f_object)
                        self.kb[f_subject] : List = []
                        self.kb[f_subject].append(' '.join([f_predicate,f_object]))
                else: # add multilanguage
                    f_subject = f_subject[3:]
                    if f_subject in self.kb:
                        self.kb[f_subject].append(' '.join([f_predicate,f_object]))
                    else:
                        #print(f_predicate)
                        #print(f_object)
                        self.kb[f_subject] : List = []
                        self.kb[f_subject].append(' '.join([f_predicate,f_object]))
            file.close()
        return

    def load_kb_pickle(self,kb_name:str):
        with open(self.kb_folder+kb_name,'rb') as handle:
            self.kb = pickle.load(handle)
        return
    def save_kb(self, type:List[str]):
        if 'csv' in type:
            with open(self.kb_folder+'{}_{}.csv'.format(self.kb_type,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")),'w') as output_file:
                for key in self.kb.keys():
                    output_file.write('{}: {}\n'.format(key,'\t'.join(s for s in self.kb[key])))
        if 'pickle' in type:
            with open(self.kb_folder+'{}_{}.pickle'.format(self.kb_type,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")),'wb') as handle:
                pickle.dump(self.kb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def get_lowercase_entity(self):
        return {key.lower():'' for key in self.kb.keys()}

    def match_entity(self, name: str ):
        if name in self.kb:
            return True
        else:
            return False

    def get_desc(self,m):
        return self.kb[m]

    def get_embed(self, desc_sents: List[str]):
        pass

    #def calcu_token_completeness

if __name__=='main':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", default="/home/nelson/Data/auto_database_foundation/datasets/YAGO/")
    args = argparser.parse_args()
    yago = YAGO_data(args)
    yago.load_kb_tsv('yagoSimpleTypes.tsv')
    yago.save_kb(['csv','pickle'])

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder", default="/home/nelson/Data/auto_database_foundation/datasets/YAGO/")
args = argparser.parse_args()


#yago_simple = YAGO_data(args,'Simple')
#yago_simple.load_kb_pickle('yago_simple_multi_lingual/KB_2019_04_16_14_06.pickle')
#keys_simple = yago_simple.get_lowercase_entity()
#yago.load_kb_tsv('yagoSimpleTypes.tsv')
#kb_keys = yago.get_lowercase_entity()
#yago.save_kb(['csv','pickle'])


yago_facts = YAGO_data(args,'Facts')
#yago_facts.load_kb_tsv('yagoFacts.tsv')
yago_facts.load_kb_pickle('yago_facts/Facts_2019_04_16_21_22.pickle')
keys_facts = yago_facts.get_lowercase_entity()
#yago_facts.save_kb(['csv','pickle'])

yago_types = YAGO_data(args,'Types')
#yago_types.load_kb_tsv('yagoTypes.tsv')
yago_types.load_kb_pickle('yago_types/Types_2019_04_16_14_41.pickle')
keys_types = yago_types.get_lowercase_entity()
#yago_types.save_kb(['csv','pickle'])

yago_date = YAGO_data(args,'Date')
#yago_date.load_kb_tsv('yagoDateFacts.tsv')
yago_date.load_kb_pickle('yago_date/Date_2019_04_16_21_44.pickle')
keys_date = yago_date.get_lowercase_entity()
#yago_date.save_kb(['csv','pickle'])

corpus = NLPTaskDataFetcher.load_corpus(task=NLPTask['TAC'], files=filenames_conll)
#corpus_dict = corpus.make_vocab_dictionary()

sentences = corpus.get_all_sentences()
m_found = 0
m_miss = 0
for sentence in sentences:
    spans = [tag.text for tag in sentence.get_spans('ner')]
    for span in spans:
        if span.lower() in keys_types or span.lower() in keys_facts or span.lower() in keys_date:
            m_found += 1
        else:
            m_miss +=1

print("m_found: {}\tm_miss: {}\tfound rate: {}".format(str(m_found),str(m_miss),str(m_found/(m_found+m_miss))))

