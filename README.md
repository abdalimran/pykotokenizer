# PyKoTokenizer
PyKoTokenizer is a Korean text tokenizer for Korean Natural Language Processing tasks. It includes deep learning (RNN) model-based word tokenizers as well as morphological analyzer based word tokenizers for Korean language.

## Segmentation of Korean Words
Written Korean texts do employ white space characters. However, more often than not,
Korean words occur in a text concatenated immediately to adjacent words without
an intervening space character. This low degree of separation of words from
each other in writing is due somewhat to an abundance of what linguists call "endoclitics" 
in the language.

As the language has been subjected to principled and rigorous study for a few
decades, the issue of which strings of sounds, or letters, are words and which are
not, has been settled among a small group of selected linguists. This kind of
advancement has not been propagated to the general public yet, and nlp
engineers working on Korean cannot but make do with whatever inconsistent
grammars they happen to have access to. Thus, a major source of difficulty in
developing competent Korean text processors has been, and still is, the notion of a word
as the smallest syntactic unit.

## How to install
Before using this package please make sure you have the following dependencies installed in your system.
* **Python >= 3.6**
* **numpy >= 1.19.0**
* **pandas >= 1.1.5**
* **tensorflow >= 2.6.2**
* **h5py >= 3.1.0**
* **konlpy >= 0.5.2**

Use the following command to install the package:
```python
pip install pykotokenizer
```

## How to Use

## Model-based Tokenizers
Below, we show examples of using model-based tokenizers.

### Using KoTokenizer

```python
from pykotokenizer import KoTokenizer

tokenizer = KoTokenizer()

korean_text = "김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다."

tokenizer(korean_text)
```

**Output:**
```
"김 형호 영화 시장 분석가 는 ' 1987 ' 의 네이버 영화 정보 네티즌 10 점평 에서 언급 된 단어 들 을 지난 해 12 월 27 일 부터 올해 1 월 10 일 까지 통계 프로그램 R 과 KoNLP 패키지 로 텍스트 마이닝 하여 분석 했다 ."
```

### Using KoSpacing

```python
from pykotokenizer import KoSpacing

spacing = KoSpacing()

korean_text = "김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다."

spacing(korean_text)
```

**Output:**
```
"김형호 영화시장 분석가는 '1987'의 네이버 영화 정보 네티즌 10점 평에서 언급된 단어들을 지난해 12월 27일부터 올해 1월 10일까지 통계 프로그램 R과 KoNLP 패키지로 텍스트마이닝하여 분석했다."
```

## Morphological analyzer based Tokenizers

Below, we show examples of using morphological analyzer based tokenizers. These tokenizers has ***dependency*** on **[KoNLPy](https://konlpy.org/en/latest/)**. So, please install KoNLPy before using these. To install KoNLPy please visit this link - [https://konlpy.org/en/latest/install/](https://konlpy.org/en/latest/install/) and follow the procedure. KoNLPy requires Java in your system.

### Using KoKkma

```python
from pykotokenizer import KoKkma

kokkma = KoKkma()

korean_text = "김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다."

kokkma(korean_text)
```

**Output:**
```
"김 형 호 영화 시장 분석가 는 ' 1987 ' 의 네이버 영화 정보 네티즌 10 점 평 에서 언급 되 ㄴ 단어 들 을 지난해 12 월 27 일 부터 올해 1 월 10 일 까지 통계 프로그램 R 과 KoNLP 패키지 로 텍스트 마이닝 하 여 분석 하 었 다 ."
```

### Using KoKomoran

```python
from pykotokenizer import KoKomoran

kokomoran = KoKomoran()

korean_text = "김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다."

kokomoran(korean_text)
```

**Output:**
```
"김형호 영화 시장 분석가 는 ' 1987 ' 의 네이버 영화 정보 네티즌 10 점 평 에서 언급 되 ㄴ 단어 들 을 지난해 12월 27 일 부터 올해 1월 10 일 까지 통계 프로그램 R 과 KoNLP 패키지 로 텍스트 마 이닝 하 아 분석 하 았 다 ."
```

## Credits
This package is a revamped and customized version of the following two sources:
* KoTokenizer: https://pypi.org/project/hangul-korean/
* KoSpacing: https://github.com/haven-jeon/PyKoSpacing
* KoNLPy: https://konlpy.org/en/latest/