from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas import read_csv
import pkg_resources
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class KoTokenizer:

    tags = ['0', '1', '2']
    tag2idx = {t: i for i, t in enumerate(tags)}
    tag2idx['充'] = 2

    def __init__(self):
        modelFile = pkg_resources.resource_filename(
            'pykotokenizer', os.path.join('resources', 'ko_tokenizer', 'models', 'model_th7.hdf5'))
        word2idxFile = pkg_resources.resource_filename(
            'pykotokenizer', os.path.join('resources', 'ko_tokenizer', 'data', 'word2idx0.json'))
        self._model = load_model(modelFile)
        self._layer = self._model.get_layer(index=0)
        self._max_len = self._layer.output_shape[1]
        with open(word2idxFile, 'r') as afile:
            self.word2idx = json.load(afile)
        self._infile = None
        self._passage = None
        now = datetime.now()
        suffix = now.strftime("_%y%j_%H%M")
        self._outfile = "segmented" + suffix + ".txt"
        self._testli = list()
        self.segmented_text = ''

    def infile(self, fileName):
        self._infile = fileName

    def outfile(self, fileName):
        self._outfile = fileName

    def input_as_string(self, aStr):
        self._passage = aStr

    @property
    def testli(self):
        return self._testli

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, v):
        self._max_len = v

    @property
    def passage(self):
        return self._passage

    @staticmethod
    def letters_nonoffs(seq):
      simpleseq = []
      onoff = []
      space = True
      for i in range(len(seq)):
        ch = seq[i]
        if ch == ' ':
            space = True
            continue
        if space:
            onoff.append('1')
        else:
            onoff.append('0')
        simpleseq.append(ch)
        space = False
        if not len(simpleseq) == len(onoff):
          print("Calculation seriously flawed! Stop processing!")
          print("simpleseq", str(len(simpleseq)))
          print("onoff", str(len(onoff)))
          print(seq)
          break
      return simpleseq, onoff

    @staticmethod
    def cut_into_pieces(kul, lnth):
        ans = [0]
        last = 0
        j = 1
        kullen = len(kul)
        for i in range(0, kullen):
            if kul[i] == ' ':
                if i > j * lnth - 1:
                    if ans[-1] != last:
                        ans.append(last)
                    j += 1
                last = i
        if i > j * lnth - 1:
            ans.append(last)
        ans.append(kullen)
        return ans

    @staticmethod
    def halve(aLine):
        kili = len(aLine) // 2 + 1
        return [aLine[:kili], aLine[kili:]]

    @staticmethod
    def halve_lines(lis, threshold):
        lis2 = list()
        for ln in lis:
            if len(ln) > threshold:
                lis2.extend(KoTokenizer.halve(ln))
            else:
                lis2.append(ln)
        return lis2

    def do_segment(self):
        self._testli = list()
        if self._infile == None:
            if self.passage == None:
                sys.exit("\tOops! No text has been given to the tokenizer!")
            points = KoTokenizer.cut_into_pieces(self.passage, self.max_len)
            for i in range(0, len(points) - 1):
                st = points[i]
                end = points[i + 1]
                ls, oos = KoTokenizer.letters_nonoffs(self.passage[st:end])
                self.testli.append(list(zip(ls, oos)))
            self.segmented_text = self.segment_proper(self.testli)
        else:
            df = read_csv(self._infile, header=None, sep="\n",
                          encoding="utf-8", error_bad_lines=False)
            for _, j in df.iterrows():
                points = KoTokenizer.cut_into_pieces(j[0], self.max_len)
                for i in range(0, len(points) - 1):
                    st = points[i]
                    end = points[i + 1]
                    ls, oos = KoTokenizer.letters_nonoffs(j[0][st:end])
                    self.testli.append(list(zip(ls, oos)))
            self.segmented_text = self.segment_proper(self.testli)
            with open(self._outfile, "w") as outFile:
                outFile.write(self.segmented_text)

    def reconstruct(self, dfarr):
        aStr = ''
        for i in range(len(dfarr)):
            b = dfarr[i]
            for j in range(min(self.max_len - 1, len(self.testli[i]))):
                ch = self.testli[i][j][0]
                if b[j][KoTokenizer.tag2idx['1']] > 0.5:
                    aStr += ' '
                aStr += ch
            aStr += '\n'
        return aStr

    def segment_proper(self, aList):
        sentences = aList[:]
        maxlen = max([len(s) for s in sentences])
        while maxlen > self.max_len:
            sentences = KoTokenizer.halve_lines(sentences, self.max_len)
            maxlen = len(sentences)

        X = list()
        for s in sentences:
            anX = list()
            for w in s[:self.max_len]:
                val = self.word2idx.get(w[0])
                if val is None:
                    # assimilate alien characters to a rare character
                    val = len(self.word2idx) - 1
                anX.append(val)
            X.append(anX)
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post",
                          value=self.word2idx['充'])
        arr = self._model.predict(X, verbose=0)
        return self.reconstruct(arr)

    @property
    def segmented_output(self):
        return self.segmented_text

    def __call__(self, text):
        self.input_as_string(text)
        self.do_segment()
        tokenized_sent = self.segmented_output
        return tokenized_sent.strip(' \n\t')
