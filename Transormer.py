#To run in colab: https://colab.research.google.com/drive/1d67mptZFw7RLKsEsY-HfYsznZ5IIzBpq?usp=sharing

import tensorflow as tf #Başta tensorflow olmak üzere gerekli kütüphaneler.
import tensorflow_datasets as tfds
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import re
import sys
from google.cloud import translate_v2 as translate
from tkinter import *
import tkinter.messagebox

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"/GoogleCloudKey_owneracc.json"  #TRANSLATE API BAĞLANTISI İÇİN KEY DOSYASI
translate_client = translate.Client()
target = 'tr'

try: #Kodun TPU üzerinde çalışmasını sağlayan kod parçası.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()



#Attention Is All You Need makalesindeki optimize parametreler.
MAX_LENGTH = 40              
MAX_SAMPLES = 50000
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
BUFFER_SIZE = 20000
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 100

#Verimizi harf ve noktalama işaretleri dışındaki karakterlerden temizliyoruz.
#ve bütün karakterleri küçük hale getiriyoruz.
def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = re.sub(r"i'm", "i am", sentence)
  sentence = re.sub(r"he's", "he is", sentence)
  sentence = re.sub(r"she's", "she is", sentence)
  sentence = re.sub(r"it's", "it is", sentence)
  sentence = re.sub(r"that's", "that is", sentence)
  sentence = re.sub(r"what's", "that is", sentence)
  sentence = re.sub(r"where's", "where is", sentence)
  sentence = re.sub(r"how's", "how is", sentence)
  sentence = re.sub(r"\'ll", " will", sentence)
  sentence = re.sub(r"\'ve", " have", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"\'d", " would", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"won't", "will not", sentence)
  sentence = re.sub(r"can't", "cannot", sentence)
  sentence = re.sub(r"n't", " not", sentence)
  sentence = re.sub(r"n'", "ng", sentence)
  sentence = re.sub(r"'bout", "about", sentence)
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  return sentence

#Datamızı input ve output olarak iki tane ayrı diziye dolduruyoruz.
def get_reddit_data():
    with open("/reddit_data.txt", "r", encoding='utf-8') as file:
        sentences = file.readlines()
    inputs = []
    outputs = []
    for i in range(len(sentences) - 1):
        input = preprocess_sentence(sentences[i])
        output = preprocess_sentence(sentences[i+1])
        inputs.append(input)
        outputs.append(output)
    return inputs, outputs

questions, answers = get_reddit_data()

#Tüm veriyi token'lere çevirmek için bir tokenizer oluşturuyoruz

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

VOCAB_SIZE = tokenizer.vocab_size + 2

#Tokenizer ile tüm veriyi Token'lere çeviriyoruz ve dictionary haline getiriyoruz.
#Cümlelerin başı ve sonu anlaşılsın diye start_token ve end_token ekliyoruz.
#Tüm vektörler eşit boyuta ulaşması için "padding" işlemini gerçekleştiriyoruz.
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs = []
    tokenized_outputs = []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:  
          tokenized_inputs.append(sentence1)
          tokenized_outputs.append(sentence2)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

#Datamızı Tensor'un alabileceği şekle uygun hale getiriyoruz.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))
#Datamızın paralel bir şekilde işlenebilmesi ve karıştırılmasını sağlayan kod kısmı
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#Positional Encoding kısmını bu Class sayesinde tanımlıyoruz. Class olmasının sebebi
#Keras Layer'ları object olarak geçirmek istememiz.
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model) #Class çalışıyor

    def get_config(self):
        config = super(PositionalEncoding, self).get_config() #Değerleri alıyoruz
        config.update({
            'position': self.position,
            'd_model': self.d_model,

        })
        return config
    #sin ve cos için kullanılacak açı değerini hesaplayan fonksiyon
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    #positional encoding'i hesaplayan fonksiyon
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        
        sines = tf.math.sin(angle_rads[:, 0::2]) #çift değerler sin olarak hesaplanır
        cosines = tf.math.cos(angle_rads[:, 1::2]) #tek değerler cos olarak hesaplanır

        pos_encoding = tf.concat([sines, cosines], axis=-1) #Positional encoding tek bir matrise olarak birleştirilir
        print("<-------------------->")
        print(pos_encoding)
        print("<-------------------->")
        pos_encoding = pos_encoding[tf.newaxis, ...] 
        return tf.cast(pos_encoding, tf.float32) #Tensoru float'a çevirip döndürüyoruz.

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
#padding degerlerini maskeliyoruz
def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32) #x değeri 0'sa float bir mask tensor oluşturur
  print (mask[:, tf.newaxis, tf.newaxis, :])
  return mask[:, tf.newaxis, tf.newaxis, :] #o anki degeri dondurur

#kendinden sonraki degerleri maskeleme fonksiyonu
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1] #inputun şekli
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) #inputla aynı boyutta 1'lerden oluşan
  padding_mask = create_padding_mask(x)                                         #bir matris oluşturuluyor ve 
                                                                                #alt üçgen halini alıyor
  return tf.maximum(look_ahead_mask, padding_mask)  #iki matrisin buyuk olan degerleri tek bir matriste toplanır.                           

def scaled_dot_product_attention(query, key, value, mask):
  matmul_qk = tf.matmul(query, key, transpose_b=True) #query ve key matrisleri çarpılıyor
  depth = tf.cast(tf.shape(key)[-1], tf.float32) #input matrisinin boyutu alınıyor
  logits = matmul_qk / tf.math.sqrt(depth) # qk matrisi, input boyutunun köküne bölünüyor
  if mask is not None: #eğer maske varsa maske ile matmul_qk çarpılıyor
    logits += (mask * -1e9)
  attention_weights = tf.nn.softmax(logits, axis=-1) #elde edilen matrise softmax uygulanıyor
  output = tf.matmul(attention_weights, value) #son olarak elde edilen matris value ile çarpılıyor
  return output

class MultiHeadAttention(tf.keras.layers.Layer): 

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads #head sayısı
        self.d_model = d_model #input boyutu
        self.depth = d_model // self.num_heads 

        self.query_dense = tf.keras.layers.Dense(units=d_model) #layer yaratmak için kullanılacak objeler
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)
  

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        })
        return config

    def split_heads(self, inputs, batch_size):                        
        inputs = tf.keras.layers.Lambda(lambda inputs: tf.reshape(     
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)))(inputs)
        return tf.keras.layers.Lambda(lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3]))(inputs)

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask'] #input tensorunden Q, K, V, ve Mask değerleri alınıyor
        batch_size = tf.shape(query)[0] #query ile aynı şekilde bir tensor oluşturuluyor

        query = self.query_dense(query) #q, k ve v layer'ları oluşturuluyor
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size) #oluşturulan layer'lar head'e yollanıyor
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask) #scaled dot product attention
        scaled_attention = tf.keras.layers.Lambda(lambda scaled_attention: tf.transpose( #hesaplanıyor
            scaled_attention, perm=[0, 2, 1, 3]))(scaled_attention)

        concat_attention = tf.keras.layers.Lambda(lambda scaled_attention: tf.reshape(scaled_attention,
                                                                                      (batch_size, -1, self.d_model)))(
            scaled_attention) #tüm attention'lar birleştiriliyor

        outputs = self.dense(concat_attention) #son katman oluşturuluyor

        return outputs

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs") #girdi tensoru oluşturuluyor
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask") #mask tensoru olusturuluyor

  attention = MultiHeadAttention( #Multi head'e gerekli değerler gönderiliyor
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention) #Dropout uyguluyoruz
  add_attention = tf.keras.layers.add([inputs,attention]) #attention ile input'lar birleştiriliyor
  attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention) #normalizasyon işlemi yapılıyor

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention) #layer oluşturuluyor ve veri relu'dan geçiriliyor
  outputs = tf.keras.layers.Dense(units=d_model)(outputs) #d_model boyutunda layer sıkıştırılıyor
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs) #dropout uygulanıyor
  add_attention = tf.keras.layers.add([attention,outputs]) #attention ile output toplanıyor
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention) #normalizasyon işlemi

  return tf.keras.Model( #layerlar bir encoder objesine çeviriliyor
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
#encoder block'u oluşturuluyor
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs") #input tensoru oluşturuluyor
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask") #maske tensoru oluşturuluyor

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs) #input değerleri embedding bir vektore donusturuluyor
  embeddings *= tf.keras.layers.Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model)
  embeddings = PositionalEncoding(vocab_size,d_model)(embeddings) #positional encoding islemi yapılıyor

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings) #outputs dropout oluşturuluyor

  for i in range(num_layers):
    outputs = encoder_layer( #num_layers kadar layer oluşturuluyor
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model( #bütün layerlar bir encoder objesine çeviriliyor
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs") #inputs tensor oluşturuluyor
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs") #encoder'dan gelen değer için tensor olusturuluyor
  look_ahead_mask = tf.keras.Input( #bir sonraki değerleri engelleyen maske objesi oluşturuluyor                 
      shape=(1, None, None), name="look_ahead_mask") 
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask') #padding maskesi oluşturuluyor

  attention1 = MultiHeadAttention( #multi head attention layer'ları oluşturuluyor
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  add_attention = tf.keras.layers.add([attention1,inputs]) #attention degeri ekleniyor
  attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention) #normalizasyon işlemi

  attention2 = MultiHeadAttention( #multi head attention layer'ları oluşturuluyor
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2) #dropout uygulanıyor
  add_attention = tf.keras.layers.add([attention2,attention1]) #attention degeri ekleniyor
  attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention) #normalizasyon işlemi

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2) #çıktılar için layer oluşturuluyor , relu uygulanıyor
  outputs = tf.keras.layers.Dense(units=d_model)(outputs) #d_model'e sıkıştırılıyor
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs) #dropout uygulanıyor
  add_attention = tf.keras.layers.add([outputs,attention2]) #attention değeri ekleniyor
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_attention) #normalizasyon uygulanıyor

  return tf.keras.Model( #decoder layer oluşturuluyor
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder(vocab_size, #decoder bloğu oluşturuluyor
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs') #input tensor
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs') #encoder'dan gelen değer için tensor
    look_ahead_mask = tf.keras.Input( #kendinden sonraki kelimeler için değerler maske tensoru
        shape=(1, None, None), name='look_ahead_mask') 
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask') #padding maske tensoru

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs) #input embedding layer'ı haline getiriliyor
    embeddings *= tf.keras.layers.Lambda(lambda d_model: tf.math.sqrt(tf.cast(d_model, tf.float32)))(d_model) 
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings) #positional encoding islemi ile çarpılıyor.

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings) #dropout işlemi yapılıyor

    for i in range(num_layers): #num_layer kadar decoder_layer oluşturuluyor
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model( #decoder bloğu
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def transformer(vocab_size, #transformer oluşturuluyor
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs") #input tensoru
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs") #bir önceki output tensoru

  enc_padding_mask = tf.keras.layers.Lambda( #padding değerlerini etkisiz hale getiriyoruz
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
 
  look_ahead_mask = tf.keras.layers.Lambda( #decoder'a verilen değerlerin kendinden sonraki değerlerden 
      create_look_ahead_mask,               #etkilenmemesi için maske       
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  dec_padding_mask = tf.keras.layers.Lambda( #padding değerlerini etkisiz hale getiriyoruz
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder( #encoder oluşturuluyor
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder( #decoder oluşturuluyor
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name) #transformer modeli döndürülüyor
 
sample_transformer = transformer( #transformer oluşturuluyor
    vocab_size=8192,
    num_layers=4,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_transformer")


def loss_function(y_true, y_pred): #loss değerini ölçüyoruz
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1)) #

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule): #learning_rate ölçülüyor

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.constant(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.multiply(tf.math.rsqrt(self.d_model), tf.math.minimum(arg1, arg2))

sample_learning_rate = CustomSchedule(d_model=128) 

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32))) #learning rate graifiği
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

tf.keras.backend.clear_session() #her şeyi sıfırla

learning_rate = CustomSchedule(D_MODEL) #learning rate hesaplanıyor

optimizer = tf.keras.optimizers.Adam( #adam optimizer oluşturuluyor
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred): #accuracy değerleri ölcülüyor
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

#TPU ile transformer başlatılıyor
with strategy.scope():
  model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

  model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy]) #modeli çalıştırıyoruz
model.fit(dataset, epochs=EPOCHS) #modele dataseti oluşturuyoruz

def evaluate(sentence): #gelecek kelimeler hesaplanıyor
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence): #TAHMİN İŞLEMLERİ YÜRÜTÜLÜYOR
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])
  output1 = translate_client.translate(
    sentence,
    target_language = target)
  output2 = translate_client.translate(
    predicted_sentence,
    target_language = target)
  print('Input: {}'.format(output1['translatedText']))
  print('Output: {}'.format(output2['translatedText']))
  return predicted_sentence
  #print('Input: {}'.format(sentence)) ----->İNGİLİZCE SONUÇLAR İÇİN
  #print('Output: {}'.format(predicted_sentence)) ----->İNGİLİZCE SONUÇLAR İÇİN

"""class TkinterApp(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        Tk.wm_title(self, "Chatbot v1.0")
        # creating a container
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Manual, Auto):
            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, background="LightSteelBlue4")

        heading = Label(self, background="LightSteelBlue4", text="Welcome to Start Page", font=('arial 40 bold'),
                        fg='black')
        heading.place(x=250, y=0)

        menu = Label(self, background="LightSteelBlue4", text="Menu", font="arial 18 bold")
        menu.place(x=10, y=0)

        info = Label(self, background="LightSteelBlue4",
                     text="If you want to enter input, click to Manual button\nIf you want everything automatic, click Auto Button",
                     anchor="e", justify=LEFT, font="arial 18 bold")
        info.place(x=220, y=70)

        button1 = Button(self, highlightbackground="LightSteelBlue4", text="Manual",
                         command=lambda: controller.show_frame(Manual))
        button1.place(x=0, y=30)

        button2 = Button(self, highlightbackground="LightSteelBlue4", text="Auto",
                         command=lambda: controller.show_frame(Auto))
        button2.place(x=0, y=60)

class Manual(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, background="LightSteelBlue4")

        heading = Label(self, background="LightSteelBlue4", text="Manual Run", font='arial 40 bold', fg='black')
        heading.place(x=250, y=0)

        menu = Label(self, background="LightSteelBlue4", text="Menu", font="arial 18 bold")
        menu.place(x=10, y=0)

        name_l = Label(self, background="LightSteelBlue4", text="Enter input", font='arial 18 bold')
        name_l.place(x=150, y=70)

        self.name_e = Entry(self, highlightbackground="LightSteelBlue4", width=25, font=('arial 18 bold'))
        self.name_e.place(x=380, y=70)

        manInput = self.name_e.get()
        manInput = str(manInput)

        tBox = Text(self, highlightbackground="LightSteelBlue4", relief=RIDGE, width=50, height=24, borderwidth=2)
        tBox.place(x=380, y=110)

        def predictConnector(self):
            predSent = predict(self)
            tBox.insert(END, "Chatbot: " + str(predSent) + "\n")

        btn_add = Button(self, highlightbackground="LightSteelBlue4", text="Start", width=15, height=2, bg='blue',
                         fg='white', command=lambda: predictConnector(manInput))
        btn_add.place(x=650, y=65)

        button1 = Button(self, highlightbackground="LightSteelBlue4", text="StartPage",
                         command=lambda: controller.show_frame(StartPage))
        button1.place(x=0, y=30)

        button2 = Button(self, highlightbackground="LightSteelBlue4", text="Auto",
                         command=lambda: controller.show_frame(Auto))
        button2.place(x=0, y=60)

class Auto(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, background="LightSteelBlue4")

        heading = Label(self, background="LightSteelBlue4", text="Auto Run", font='arial 40 bold', fg='black')
        heading.place(x=250, y=0)

        menu = Label(self, background="LightSteelBlue4", text="Menu", font="arial 18 bold")
        menu.place(x=10, y=0)

        name_l = Label(self, background="LightSteelBlue4", text="Enter conversation number", font='arial 18 bold')
        name_l.place(x=140, y=70)

        self.name_e = Entry(self, highlightbackground="LightSteelBlue4", width=25, font=('arial 18 bold'))
        self.name_e.place(x=380, y=70)
        def predictConnector(self):
            for _ in range(self):
                predSent = predict("What is your name")
                tBox.insert(END, "Chatbot: " + str(predSent) + "\n")

        def inpGetter(self):
            autoInput = self.name_e.get()
            autoInput1 = int(autoInput)
            predictConnector(autoInput1)

        tBox = Text(self, highlightbackground="LightSteelBlue4", relief=RIDGE, width=50, height=24, borderwidth=2)
        tBox.place(x=380, y=110)

        btn_add = Button(self, highlightbackground="LightSteelBlue4", text="Start", width=15, height=2, bg='blue',
                         fg='white', command=lambda: inpGetter(self))
        btn_add.place(x=650, y=65)

        button1 = Button(self, highlightbackground="LightSteelBlue4", text="StartPage",
                         command=lambda: controller.show_frame(StartPage))
        button1.place(x=0, y=30)

        button2 = Button(self, highlightbackground="LightSteelBlue4", text="Manual",
                         command=lambda: controller.show_frame(Manual))
        button2.place(x=0, y=60)

app = TkinterApp()
app.geometry("1100x600+0+0")
app.resizable(width="FALSE", height="FALSE")
app.mainloop()
    """  


while(true):
  input(get)
  if input(get) != "exit":
    predict(get)
  else:
    exit(0)
