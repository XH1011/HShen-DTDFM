import tensorflow as tf
import pickle5 as pickle
import numpy as np
from sklearn.preprocessing import normalize
from Utils.utils import ZCA

# Tensorflow preprocessing operations.
def preprocess(x):
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))
def preprocess_cwru(x):
    return normalize(x)

def load_data_zca(x0,BATCH_SIZE):

    x0=x0.astype(np.float32)
    trf = ZCA().fit(x0)
    x_zca = trf.transform(x0)#whitened
    x_hat = trf.inverse_transform(x_zca)
    print(np.allclose(x0,x_hat))
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,trf,x0,x_hat,x_zca

def LoadData_pickle(path,type='rb'):
  with open(path+'.pkl', type) as f:
      print(f)
      data = pickle.load(f)
  return data
def Get_index(Datasets):
    index_list = []
    for i in range(106):
        index = int(Datasets.all_labels[i][1])
        index_list.append(index)
    index_list = np.array(index_list)
    print(index_list)
    print(Datasets.all_labels)
    return index_list
def get_test(Datasets,fault_index):
    X_test=Datasets.X_test
    y_test=np.array(list(map(int,Datasets.y_test)))
    # print(y_test)
    number=[]
    X_T=[]
    for j in fault_index:
        pzq=X_test[np.where(y_test==j)].shape[0]
        X = X_test[np.where(y_test == j)][:,:,0]
        X_T.append(X)
        number.append(pzq)

    print(len(number),sum(number))
    return X_T
def get_train(Datasets,fault_index):
    X_train=Datasets.X_train
    y_train=np.array(list(map(int,Datasets.y_train)))
    print(y_train)
    number2=[]
    X_t=[]
    for j in fault_index:
        pzq=X_train[np.where(y_train==j)].shape[0]
        X = X_train[np.where(y_train == j)][:,:,0]
        X_t.append(X)
        number2.append(pzq)

    print(len(number2),sum(number2))
    return X_t


def load_data(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0 = pickle.load(f)[0]
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,x0


