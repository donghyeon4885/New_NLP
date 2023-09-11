import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'],
                           ['barber', 'huge', 'person'], ['knew', 'secret'], 
                           ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], 
                           ['barber', 'kept', 'word'], ['barber', 'kept', 'word'],
                            ['barber', 'kept', 'secret'], 
                            ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
                            ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

max_len = max(len(item) for item in encoded)
print('최대 길이 :',max_len)

# 길이가 7보다 짧은 문장에는 숫자 0을 채워서 길이 7로 맞춘다. ==> 제로패딩
for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

padded = pad_sequences(encoded)
padded # pad_sequences는 기본적으로 문서의 뒤에 0을 채우지않고 앞에 0을 채운다. 인자로  padding='post'을 추가해주면 해결됨.

padded = pad_sequences(encoded, padding='post')
padded



