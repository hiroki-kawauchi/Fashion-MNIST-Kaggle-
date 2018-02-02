
# coding: utf-8

# "Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes."<https://www.kaggle.com/zalando-research/fashionmnist>  
# ### ※Labels are;
# * 0 T-shirt/top
# * 1 Trouser
# * 2 Pullover
# * 3 Dress
# * 4 Coat
# * 5 Sandal
# * 6 Shirt
# * 7 Sneaker
# * 8 Bag
# * 9 Ankle boot 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import chainer 
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer import serializers


# In[3]:


mnist_train = pd.read_csv('fashion_mnist_train.csv').values


# In[4]:


X_1 = mnist_train[:, 1:].reshape(mnist_train.shape[0],1,28, 28).astype( 'float32' )
y_1 = mnist_train[:, 0].astype(np.int32)


# In[5]:


print(X_1)
print(y_1)


# In[6]:


#画像の正規化
X_1 /= 255.


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=0)


# In[8]:


print(X_train.shape)
print(y_train.shape)


# In[9]:


# CNNの定義
class CNN(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=1, out_channels=16, ksize=3, stride=1)
            self.conv2 = L.Convolution2D(in_channels=16, out_channels=32, ksize=3, stride=1)
            self.conv3 = L.Convolution2D(in_channels=32, out_channels=32, ksize=3, stride=1) # in_channelをNoneで省略
            self.fc4 = L.Linear(None, 30)
            self.fc5 = L.Linear(30, 10) # out_channelはクラス数と同じ10に設定
        
        
    def __call__(self, X):
        h = F.leaky_relu(self.conv1(X), slope=0.05)
        h = F.leaky_relu(self.conv2(h), slope=0.05)
        h = F.leaky_relu(self.conv3(h), slope=0.05)
        h = F.leaky_relu(self.fc4(h), slope=0.05)
        return self.fc5(h)


# In[10]:


# 必要なライブラリを読み込み
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from chainer import optimizers, serializers, training, iterators


# In[11]:


# 分類器インスタンスの生成
model = L.Classifier(CNN())

# optimizerの生成
optimizer = chainer.optimizers.SGD() # 今回はSGDを採用
optimizer.setup(model)               # モデルの構造を読み込ませる

# ミニバッチに含まれるサンプル数を指定
batchsize = 32

# epoch数を指定
n_epoch = 10


# In[12]:


# trainerを定義
train = tuple_dataset.TupleDataset(X_train,y_train)
train_iter = iterators.SerialIterator(train,batch_size=batchsize,shuffle=True)
updater = training.StandardUpdater(train_iter,optimizer)
trainer = training.Trainer(updater,(n_epoch,'epoch'),out = 'result')


# In[13]:


# Extensionsを利用してtrainerの機能を拡張
test = tuple_dataset.TupleDataset(X_test,y_test)
test_iter = iterators.SerialIterator(test,batch_size=batchsize,shuffle=False,repeat=False)
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                          'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))


# In[14]:


# 学習を実行
trainer.run()

