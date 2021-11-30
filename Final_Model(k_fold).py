#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats


# In[2]:


data = pd.read_csv('FinalData_400.csv')
data.head()


# In[3]:


final_data = data.rename(columns = {'# Label':'act',' Subject':'subject',' Ac_Hor':'ah',
                                    ' Ac_Ver':'av',' Gy_Hor':'gh',' Gy_Ver':'gv'}).copy()
# final_data.head()


# In[4]:


label = LabelEncoder()                       # labelling the float type label to integers

final_data['label'] = label.fit_transform(final_data['act'])
final_data.head()


# In[5]:


final_data['ah'] = final_data['ah'].astype(float)         # convert the data to float
final_data['av'] = final_data['av'].astype(float)
final_data['gh'] = final_data['gh'].astype(float)
final_data['gv'] = final_data['gv'].astype(float)


# In[6]:


# final_data['label'].value_counts()   
# final_data['act'].value_counts()......................checking for the equivalence of classes and labels


# In[7]:


normal = final_data[final_data['act']== 1.0].head(38838).copy()                # Balancing the Dataset  
downstairs = final_data[final_data['act']== 3.0].head(38838).copy()
upstairs = final_data[final_data['act']== 2.0].head(38838).copy()


# In[8]:


bal_data = pd.DataFrame()                            
bal_data = bal_data.append([normal, upstairs, downstairs])
bal_data.shape
#bal_data.head(100)


# In[9]:


balanced_data = bal_data.drop(['act','subject'],axis = 1).copy()   # dropping the unwanted act and subject column
# balanced_data.head()


# ## Feature Scaling (Gaussian distribution)

# In[10]:


x = balanced_data[['ah', 'av', 'gh', 'gv']]       
y = balanced_data['label']

scaler = StandardScaler()
X = scaler.fit_transform(x)                     

scaled_X = pd.DataFrame(data = X, columns=['ah', 'av', 'gh', 'gv'])
scaled_X['label'] = y.values

scaled_X


# ## Extracting frames of 4 seconds

# In[11]:


sampling_frequency = 50
frame_size = sampling_frequency*4
jump_size = sampling_frequency*2

def extract_frames(data, frame_size, jump_size):

    feature_shape = 4

    frames = []
    labels = []
    for i in range(0, len(data) - frame_size, jump_size):
        x1 = data['ah'].values[i: i + frame_size]
        y1 = data['av'].values[i: i + frame_size]
        x2 = data['gh'].values[i: i + frame_size]
        y2 = data['gv'].values[i: i + frame_size]
      
        
        # Retrieve the most often used label in this segment
        label = stats.mode(data['label'][i: i + frame_size])[0][0]
        frames.append([x1, y1, x2, y2])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, feature_shape)
    labels = np.asarray(labels)

    return frames, labels


# ## Preparing the training and testing dataset

# In[12]:


X, y = extract_frames(scaled_X, frame_size, jump_size)
X.shape, y.shape


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0, stratify = y)
X_train.shape, X_test.shape


# ## Model Architecture
# 
# model = keras.models.Sequential()
# 
# model.add(keras.layers.Flatten(input_shape=X_train[0].shape))                
# 
# model.add(keras.layers.Dense(500, activation="relu",kernel_regularizer=keras.regularizers.l1(0.001)))
# keras.layers.Dropout(rate=0.4)
# model.add(keras.layers.Dense(300, activation="relu",kernel_regularizer=keras.regularizers.l1(0.001)))
# keras.layers.Dropout(rate=0.4)
# model.add(keras.layers.Dense(100, activation="relu",kernel_regularizer=keras.regularizers.l1(0.01)))
# 
# model.add(keras.layers.Dense(3, activation="softmax"))
# 

# In[29]:


from tensorflow.keras.models import model_from_json
with open('model.json','r') as file:
    model_json = file.read()
    
loaded_model = model_from_json(model_json)                            # Loading the model architecture


# In[30]:


loaded_model.load_weights('weights.h5')                               # loading the pretrained weights


# In[25]:


loaded_model.summary()


# ## Evaluating predictions using Confusion Matrix

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools


# In[31]:


predictions = loaded_model.predict(X_test)


# In[32]:


rounded_predictions = np.argmax(predictions, axis = -1)


# In[33]:


cm = confusion_matrix(y_true = y_test, y_pred = rounded_predictions)


# In[34]:


def plot_confusion_matrix(cm, classes, normalize=False,title = 'Confusion Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix without normalization')
        
    print(cm)
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0],range(cm.shape[1]))):
        plt.text(j,i,cm[i,j],
                horizontalalignment = "center",
                color = "white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')


# In[35]:


cm_plot_labels = ['normal', 'upstairs', 'downstairs']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')


# ## K-fold Cross Validation (2,5,10)

# In[14]:


def buil_ANN():
    from tensorflow.keras.models import model_from_json
    with open('model.json','r') as file:
        model_json = file.read()
    
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('weights.h5')
    loaded_model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    
    return loaded_model


# In[15]:


from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


# ###  5-fold

# In[16]:


model = KerasClassifier(build_fn = buil_ANN , epochs = 30)
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, scoring = None, cv = 5)


# In[18]:


accuracies.mean()


# ### 2-fold

# In[19]:


model = KerasClassifier(build_fn = buil_ANN , epochs = 30)
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, scoring = None, cv = 2)


# In[20]:


accuracies.mean()


# ### 10-fold

# In[21]:


model = KerasClassifier(build_fn = buil_ANN , epochs = 30)
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, scoring = None, cv = 10)


# In[22]:


accuracies.mean()

