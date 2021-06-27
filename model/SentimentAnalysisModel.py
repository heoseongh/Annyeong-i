import os
import sys
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from transformers import BertTokenizer
from transformers import TFBertModel

SEQ_LEN = 512

class DepressAnalysisModel:

    tokenizer = None
    sentiment_model = None
    mod = sys.modules[__name__]

    def __init__(self):
        warnings.filterwarnings('ignore')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.sentiment_model = self.create_sentiment_bert()
        self.sentiment_model.load_weights('./model/sentiment_model_weight.h5')
        setattr(self.mod, 'model', self.sentiment_model)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def create_sentiment_bert(self):
        warnings.filterwarnings('ignore')
        # 버트 pretrained 모델 로드
        model = TFBertModel.from_pretrained("bert-base-multilingual-cased", from_pt=True)
        # # 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
        token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
        mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
        segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
        # 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
        bert_outputs = model([token_inputs, mask_inputs, segment_inputs])

        bert_outputs = bert_outputs[1]
        sentiment_first = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
        sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
        # 옵티마이저는 간단하게 Adam 옵티마이저 활용
        sentiment_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                metrics=['sparse_categorical_accuracy'])
        return sentiment_model

    def mean_answer_label(self, *preds):
        preds_sum = np.zeros(preds[0].shape[0])
        for pred in preds:
            preds_sum += np.argmax(pred, axis=-1)
        return np.round(preds_sum/len(preds), 0).astype(int)

    def sentence_convert_data(self, data):
        # global tokenizer
        
        tokens, masks, segments = [], [], []
        token = self.tokenizer.encode(data, max_length=SEQ_LEN, padding='max_length', truncation=True)
        
        num_zeros = token.count(0) 
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros 
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

        tokens = np.array(tokens)
        masks = np.array(masks)
        segments = np.array(segments)
        return [tokens, masks, segments]

    def category_evaluation_predict(self, sentence):
        import warnings
        warnings.filterwarnings('ignore')
        cat_dict = {'0':"슬픔", '1':"우울", '2': "분노"}
        # global mod
        sentence = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…]+", "", sentence)
        sentence = re.sub("\\n+", " ", sentence)
        sentence = re.sub("\\t+", " ", sentence)
        data_x = self.sentence_convert_data(sentence)

        setattr(self.mod, 'expression', model.predict(data_x, batch_size=1))
        preds = str(self.mean_answer_label(expression).item())
        
        if preds == '0':
            return cat_dict[preds]
        elif preds == '1':
            return cat_dict[preds]
        elif preds == '2':
            return cat_dict[preds]

# def startModel(sentence): 
#     global tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
#     sentiment_model = create_sentiment_bert()
#     sentiment_model.load_weights('./model/sentiment_model_weight.h5')
#     setattr(mod, 'model', sentiment_model)

#     sentence = '좋아한다기보단 호감있는 남자가 생겼습니다 ㅠㅠ'
#     result = category_evaluation_predict(sentence)
#     return result