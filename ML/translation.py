import pandas as pd
import numpy as np
import tensorflow as tf
import unicodedata
import re
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, GRU, Masking
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

# Fuctional API로 회귀분석 구현
X = [1,2,3,4,5,6,7,8,9]  # 공부 시간
y = [11,22,33,44,53,66,77,87,95]   # 성적

inputs = Input(shape=(1,))
output = Dense(1, activation="linear")(inputs)
linear_model = Model(inputs, output)

sgd = optimizers.SGD(lr=0.01)   # 가중치 업데이트를 위한 확률적 경사하강법 객체 생성

linear_model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
linear_model.fit(X,y,epochs=300)

class LinearRegression(tf.keras.Model) : 
    def __init__(self) : # 모델의 구조와 동적을 의미
        super(LinearRegression, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(1, input_dim=1, activation="linear")
        
    # 모델이 데이터를 입력받아 예측값을 리턴하는 포워드(forward) 연산 수행
    def call(self, x) : 
        y_pred = self.linear_layer(x)
        return y_pred

model = LinearRegression()

sgd = tf.keras.optimizers.SGD(lr=0.01)   # 가중치 업데이트를 위한 확률적 경사하강법 객체 생성
model.compile(optimizer=sgd, loss="mse", metrics=["mse"])
model.fit(X,y,epochs=300)

lines = pd.read_csv("./datas/fra.txt", names=["src", "tar", "lic"], sep="\t")
del lines["lic"]
print("전체 샘플의 개수 :", len(lines))

# 60,000개의 샘플만 가지고 기계 번역기를 구축
lines = lines.loc[:, "src":"tar"]
lines = lines[:60000]

lines.tar = lines.tar.apply(lambda x : "\t " + x + " \n")
lines.sample(10)

# 문자 집합 생성
src_vocab = set()
for line in lines.src :  # 1줄 씩 읽음
    for char in line :   # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar :  # 1줄 씩 읽음
    for char in line :   # 1개의 문자씩 읽음
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print("source 문장의 char 집합 :", src_vocab_size)
print("target 문장의 char 집합 :", tar_vocab_size)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
src_to_index = dict([word, i+1] for i, word in enumerate(src_vocab))
tar_to_index = dict([word, i+1] for i, word in enumerate(tar_vocab))
print(src_to_index)
print(tar_to_index)


# 인덱스가 부여된 문자 집합으로부터 갖고 있는 훈련 데이터에 정수 인코딩을 수행
encoder_input = []
for line in lines.src : # 문장 단위로 처리하기 위한 반복문
    encoded_line = []
    for char in line : # 문자 단위로 처리하기 위한 반복문
        encoded_line.append(src_to_index[char])
        
    encoder_input.append(encoded_line)
    
print("source 문장의 정수 인코딩 :", encoder_input[:5])

# 인덱스가 부여된 문자 집합으로부터 갖고 있는 훈련 데이터에 정수 인코딩을 수행
decoder_input = []
for line in lines.tar : # 문장 단위로 처리하기 위한 반복문
    decoded_line = []
    for char in line : # 문자 단위로 처리하기 위한 반복문
        decoded_line.append(tar_to_index[char])
        
    decoder_input.append(decoded_line)
    
print("target 문장의 정수 인코딩 :", decoder_input[:5])

# 디코더의 예측값과 비교하기 위한 실제값에 대해서 정수 인코딩을 수행
decoder_target = []
for line in lines.tar : 
    timestep = 0
    encoded_line = []
    for char in line : 
        if timestep > 0 : 
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
    
print("target 문장 레이블의 정수 인코딩 :", decoder_target[:5])

# 패딩을 위한 영어, 프랑스 문장 샘플 길이 확인
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print("source 문장의 최대 길이 :", max_src_len)
print("target 문장의 최대 길이 :", max_tar_len)

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding="post")
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding="post")
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding="post")
# 원-핫 인코딩
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

# 훈련 과정에서, 이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로
# 넣어주지 않고, 이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 넣어주는
# 방법(교사강요) 사용하여 seq2seq 모델을 설계
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True) # 인코더의 내부 상태 리턴
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)  # 인코더 출력, 은닉상태, 셀 상태 리턴
encoder_states = [state_h, state_c]  # 컨텍스트 벡터

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# 디코더도 은닉 상태, 셀 상태를 리턴하지만 훈련 과정에서는 사용하지 않습니다
decoder_softmax_layer = Dense(tar_vocab_size, activation="softmax")

# 출력층에 프랑스어의 단어 집합의 크기만큼 뉴런을 배치한 후
# 소프트 맥스 함수를 사용하여 실제값과 오차를 구합니다.
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64,
         epochs=40, validation_split=0.2)

# 훈련 과정에서, 이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로
# 넣어주지 않고, 이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 넣어주는
# 방법(교사강요) 사용하여 seq2seq 모델을 설계
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True) # 인코더의 내부 상태 리턴
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)  # 인코더 출력, 은닉상태, 셀 상태 리턴
encoder_states = [state_h, state_c]  # 컨텍스트 벡터

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# 디코더도 은닉 상태, 셀 상태를 리턴하지만 훈련 과정에서는 사용하지 않습니다
decoder_softmax_layer = Dense(tar_vocab_size, activation="softmax")

# 출력층에 프랑스어의 단어 집합의 크기만큼 뉴런을 배치한 후
# 소프트 맥스 함수를 사용하여 실제값과 오차를 구합니다.
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64,
         epochs=40, validation_split=0.2)