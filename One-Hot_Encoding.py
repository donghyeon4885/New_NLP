# 원핫 인코딩이란?
# 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여, 다른 인덱스에는 0을 부여.
# 1. 정수 인코딩 
# 2. 표현하고 싶은 단어의 고유한 정수를 인덱스로 간주하고 해당 위치에 1을 부여하고, 다른 단어에는 0을 부여.

# ex) 나는 자연어 처리르 배운다.

from konlpy.tag import Okt  

okt = Okt()  
tokens = okt.morphs("나는 자연어 처리를 배운다")  
print(tokens)

word_to_index = {word : index for index, word in enumerate(tokens)}
print('단어 집합 :',word_to_index)

def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding("자연어", word_to_index)

# 케라스를 이용한 원-핫 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)


# 정수 시퀀스로 변환
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded) #->  [2, 5, 1, 6, 3, 7]

one_hot = to_categorical(encoded)
print(one_hot)

# 원핫 인코딩의 한계
# -> 1.단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다.
# -> 2. 단어의 유사도를 표현하지 못한다.

