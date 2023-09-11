import nltk
nltk.download()

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :',word_tokenize
      ("Don't be fooled by the dark sounding name,Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))



##표준 토큰화 TreebankWordTokenizer 사용법
#1. 하이푼으로 구성된 단어는 하나로 유지한다.
#2. dosen't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리한다.
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer() 

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))

# 4.문장 토큰화
EX1= "IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 aaa@gmail.com로 결과 좀 보내줘. 그 후 점심 먹으러 가자."
EX2= "Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year."
print(tokenizer.tokenize(EX1))
print(tokenizer.tokenize(EX2))

from nltk.tokenize import sent_tokenize # 마침표를 기준으로 문장을 구분한다.

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))

import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))

## 5. 한국어에서의 토큰화의 어려움
# 한국어는 어절이 독립적인 단어로 구성되는 것이 아니라 조사등의 무언가가 붙어있는 경우가 많아서\n 
# 이를 전부 분리해줘야한다.
#-> 형태소(뜻을 가진 가장 작은 말의 단위) 토큰화를 수행해야한다.

# 6. 품사 태킹
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))

from konlpy.tag import Okt

okt = Okt()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 


