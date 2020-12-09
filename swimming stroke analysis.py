#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install keras


# In[35]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from glob import glob

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet',
               include_top=False,
               input_shape=(150,150,3))
bring=conv_base
bring.summary()


# In[2]:


import os, shutil
original_dataset_dir = 'C:\\Users\\dkenl\\OneDrive\\바탕 화면\\stroke\\train'
base_dir='C:\\Users\\dkenl\\OneDrive\\바탕 화면\\stroke\\test'


# In[3]:


if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.mkdir(base_dir)


# In[4]:


train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)


# In[5]:


test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


# In[6]:


train_back_dir = os.path.join(train_dir,'back')
os.mkdir(train_back_dir)
train_breast_dir = os.path.join(train_dir, 'breast')
os.mkdir(train_breast_dir)
train_butterfly_dir = os.path.join(train_dir, 'butterfly')
os.mkdir(train_butterfly_dir)
train_free_dir = os.path.join(train_dir, 'free')
os.mkdir(train_free_dir)

validation_back_dir = os.path.join(validation_dir, 'back')
os.mkdir(validation_back_dir)
validation_breast_dir = os.path.join(validation_dir, 'breast')
os.mkdir(validation_breast_dir)
validation_butterfly_dir = os.path.join(validation_dir, 'butterfly')
os.mkdir(validation_butterfly_dir)
validation_free_dir = os.path.join(validation_dir, 'free')
os.mkdir(validation_free_dir)


# In[7]:


fnames_back=['back ({}).jpg'.format(i) for i in range(1,251)]
fnames_breast=['breast ({}).jpg'.format(i) for i in range(1,251)]
fnames_butterfly=['butterfly ({}).jpg'.format(i) for i in range(1,251)]
fnames_free=['free ({}).jpg'.format(i) for i in range(1,251)]


for fname in fnames_back:
    src= os.path.join(original_dataset_dir,fname)
    #맨위에 사진들 있는 경로가 original
    dst = os.path.join(train_back_dir, fname)
    shutil.copyfile(src, dst)
    ##original꺼를 shutile.copyfile이용해서 복사함
for fname in fnames_breast:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_breast_dir,fname)
    shutil.copyfile(src,dst)
for fname in fnames_butterfly:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_butterfly_dir,fname)
    shutil.copyfile(src,dst)
for fname in fnames_free:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_free_dir,fname)
    shutil.copyfile(src,dst)

fnames_back=['back ({}).jpg'.format(i) for i in range(250,351)]
fnames_breast=['breast ({}).jpg'.format(i) for i in range(250,351)]
fnames_butterfly=['butterfly ({}).jpg'.format(i) for i in range(250,351)]
fnames_free=['free ({}).jpg'.format(i) for i in range(250,351)]
for fname in fnames_back:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_back_dir, fname)
    shutil.copyfile(src, dst)
for fname in fnames_breast:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_breast_dir,fname)
    shutil.copyfile(src,dst)
for fname in fnames_butterfly:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_butterfly_dir,fname)
    shutil.copyfile(src,dst)
for fname in fnames_free:
    src= os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_free_dir,fname)
    shutil.copyfile(src,dst)


# In[8]:


from keras.preprocessing.image import ImageDataGenerator


# In[9]:


base_dir='C:\\Users\\dkenl\\OneDrive\\바탕 화면\\stroke\\test'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size=10


# In[10]:


def extract_feature(directory, sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,
                                         target_size=(150,150),
                                         batch_size=batch_size,
                                         class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

train_features,train_labels=extract_feature(train_dir,250)
validation_features,validation_labels=extract_feature(validation_dir,100)


# In[11]:


train_features=np.reshape(train_features,(250,4*4*512))
validation_features=np.reshape(validation_features,(100,4*4*512))


# In[32]:


from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, Dropout , BatchNormalization, Flatten

model=models.Sequential()
model.add(layers.Dense(256,activation='relu', input_dim=4 * 4 * 512))

          
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history=model.fit(train_features,train_labels,
                 epochs=10,
                 batch_size=batch_size,
                 validation_data=(validation_features,validation_labels))
model.summary()


# In[27]:


from keras.models import Sequential
from keras.layers import Dense, Dropout , BatchNormalization, Flatten
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam


model = Sequential()
model.add(Dense(64, input_dim=train_dir[1] , activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(196, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.compile(optimizer = 'adam',loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(train_features,train_labels,
                 epochs=10,
                 batch_size=batch_size,
                 validation_data=(validation_features,validation_labels))


# In[ ]:




