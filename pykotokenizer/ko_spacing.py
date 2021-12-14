import os
import re
import csv
import json
import numpy as np
import pkg_resources
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class KoSpacing:
    def __init__(self, rules=[]):
        model_path = pkg_resources.resource_filename(
            'pykotokenizer', os.path.join('resources', 'ko_spacing', 'models', 'ko_spacing_model'))
        dic_path = pkg_resources.resource_filename(
            'pykotokenizer', os.path.join('resources', 'ko_spacing', 'dicts', 'c2v.dic'))
        MODEL = load_model(model_path)
        MODEL.make_predict_function()
        W2IDX, _ = self.load_vocab(dic_path)
        MAX_LEN = 198
        self._model = MODEL
        self._w2idx = W2IDX
        self.max_len = MAX_LEN
        self.pattern = re.compile(r'\s+')
        self.rules = {}
        for r in rules:
            if type(r) == str:
                self.rules[r] = re.compile('\s*'.join(r))
            else:
                raise ValueError("rules must to have only string values.")

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            data = json.loads(f.read())
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
        return word2idx, idx2word

    def encoding_and_padding(self, word2idx_dic, sequences, **params):
        seq_idx = [[word2idx_dic.get(a, word2idx_dic['__ETC__'])
                    for a in i] for i in sequences]
        params['value'] = word2idx_dic['__PAD__']
        return(sequence.pad_sequences(seq_idx, **params))

    def set_rules_by_csv(self, file_path, key=None):
        with open(file_path, 'r', encoding='UTF-8') as csvfile:
            csv_var = csv.reader(csvfile)
            if key == None:
                for line in csv_var:
                    for word in line:
                        self.rules[word] = re.compile('\s*'.join(word))
            else:
                csv_var = list(csv_var)
                index = -1
                for i, word in enumerate(csv_var[0]):
                    if word == key:
                        index = i
                        break

                if index == -1:
                    raise KeyError(f"'{key}' is not in csv file")

                for line in csv_var:
                    self.rules[line[index]] = re.compile(
                        '\s*'.join(line[index]))

    def get_spaced_sent(self, raw_sent):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = self.encoding_and_padding(word2idx_dic=self._w2idx,
                                           sequences=sents_in,
                                           maxlen=200,
                                           padding='post',
                                           truncating='post')
        results = self._model.predict(mat_in)
        mat_set = results[0, ]
        preds = np.array(
            ['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        return self.make_pred_sents(raw_sent_, preds)

    def make_pred_sents(self, x_sents, y_pred):
        res_sent = []
        for i, j in zip(x_sents, y_pred):
            if j == '1':
                res_sent.append(i)
                res_sent.append(' ')
            else:
                res_sent.append(i)
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs

    def apply_rules(self, spaced_sent):
        for word, rgx in self.rules.items():
            spaced_sent = rgx.sub(word, spaced_sent)
        return spaced_sent

    def __call__(self, sent):
        if len(sent) > self.max_len:
            splitted_sent = [sent[y-self.max_len:y]
                             for y in range(self.max_len, len(sent)+self.max_len, self.max_len)]
            spaced_sent = ''.join([self.get_spaced_sent(ss)
                                  for ss in splitted_sent])
        else:
            spaced_sent = self.get_spaced_sent(sent)
        if len(self.rules) > 0:
            spaced_sent = self.apply_rules(spaced_sent)
        return spaced_sent.strip()
