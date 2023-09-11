import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

X, y = zip(['a', 1], ['b', 2], ['c', 3])
print('x 데이터 :',X)
print('y 데이터 :',y)

values = [['당신에게 드리는 마지막 혜택!', 1],['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
df

X = df['메일 본문']
y = df['스팸 메일 유무']

X 데이터 : ['당신에게 드리는 마지막 혜택!', '내일 뵐 수 있을지 확인 부탁드...', '도연씨. 잘 지내시죠? 오랜만입...', '(광고) AI로 주가를 예측할 수 있다!']
y 데이터 : [1, 0, 0, 1]

print('X 데이터 :',X.to_list())
print('y 데이터 :',y.to_list())

np_array = np.arange(0,16).reshape((4,4))
print('전체 데이터 :')
print(np_array)
