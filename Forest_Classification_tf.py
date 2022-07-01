#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# In[43]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve,recall_score

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD, Adam


# In[3]:


data = pd.read_csv('cover_data.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.nunique()


# In[7]:


data.describe().T


# In[8]:


data_nonbinary = data[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points','class']]

data_nonbinary.describe().T


# 1. The average Elevation is 2959.36 and the median is 2996.0. Therefore, this is almost normal.
# 2. The median of Aspect is greater than its mean which is 155.65.
# 3. The slope is given in degrees. The meadian is small compared to the maximum, 66.0, so the distribution of slopes is clearly right skewed.
# 4. The distribution of the Horizontal_Distance_To_Hydrology is also right skewed.
# 5. Vertical_Distance_To_Hydrology values are also positively distributed. The median of the distribution is 30.0 and the maximum is 601.0.
# 6. Same as in 5.
# 7. Hillshade_3pm is approcimately normal.
# 8. Horizontal_Distance_To_Fire_Points is also positively distributed.
# 
# Most of the nonbinary features have right skewed distributions so most of the data points have features whose values are low.

# In[9]:


columns = data_nonbinary.columns
columns


# In[10]:


for col in columns:
    print(col)
    print('Skew :',round(data[col].skew(),2))
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    data[col].hist(bins=10,grid=False)
    plt.ylabel('count')
    plt.subplot(1,2,2)
    sns.boxplot(x=data[col])
    plt.show()


# Aspect is the only nonbinary variable that has no outliers. Most of the data points belong to the first 2 classes so the classes are unbalanced.

# In[11]:


plt.figure(figsize=(16,16))
corr_mat = data_nonbinary.corr()
sns.heatmap(corr_mat,annot=True)
plt.show()


# Horizontal_Distance_To_Hidrology and Vertical_Distance_To_Hydrology are positively correlated. Aspect and Hillshade_3pm are also positvely and highly correlated.
# Elevation is positively correlated to most of the variables that are used to generate the heatmap.
# Most of the variables are negatively correlated to class which means that as the corresponding value of the variables increase the class number goes down. Therefore, high class numbers are seen when most of the
# when data points are associated with low nonbinary values.

# In[12]:


data_binary_subset = data[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9','Soil_Type10','class']]
data_binary_subset


# In[13]:


corr_mat = data_binary_subset.corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr_mat,annot=True)
plt.show()


# Correlations between binary variables and classes tell us what areas and soil types favor high classes. Wilderness_Area4 is possitively correlated to many soil types, and it is also positively correlated to class. Therefore, in those areas forest cover types associated with high classes are favored.

# In[14]:


labels = data.iloc[:,-1]
labels


# In[15]:


features = data.drop(columns=['class'])


# In[16]:


training_set,test_set,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,stratify=labels,random_state=1)
print(training_set.shape)
print(test_set.shape)


# In[17]:


standard_col = data_nonbinary.drop(columns='class').columns
ct = ColumnTransformer([('Nonbinary', StandardScaler(),standard_col)],remainder='passthrough',verbose=1)
training_scaled = ct.fit_transform(training_set)
test_scaled = ct.transform(test_set)


# In[65]:


le = LabelEncoder()
y_train=le.fit_transform(labels_train.astype(int))
y_test=le.transform(labels_test.astype(int))


# In[66]:


y_train = to_categorical(y_train,dtype='int64')
y_test = to_categorical(y_test,dtype='int64')
print(y_train.shape)
print(y_test.shape)


# In[31]:


model = Sequential()
model.add(InputLayer(input_shape=(training_scaled.shape[1],)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(7,activation='softmax'))
op = Adam(learning_rate=0.001)
# 0.001 gave reasonable results.
e_stop = EarlyStopping(monitor='loss',patience=10)
model.summary()


# In[32]:


model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=op,metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.AUC()],callbacks=[e_stop])


# In[33]:


history = model.fit(training_scaled,y_train,epochs=50,batch_size = 3000,verbose=1,validation_split = 0.12)


# In[ ]:





# In[40]:


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model categorical accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc_4'])
ax2.plot(history.history['val_auc_4'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
 
fig.savefig('/Users/luisreyna/Downloads/Summer_2022/Data Science and ML Program/datasets/my_training.png')

plt.show()


# In[50]:


loss,cat_acc,acc = model.evaluate(test_scaled,y_test)


# In[51]:


print("Loss: ", loss)
print("Cat_Acc: ", cat_acc)
print("Acc: ", acc)


# In[57]:


y_pred_train = model.predict(training_scaled)
y_pred_train =np.argmax(y_pred_train, axis=1)

y_train =np.argmax(y_train, axis=1)


# In[58]:


print(y_pred_train)
print(y_train)


# In[62]:


print(classification_report(y_train, y_pred_train))

cm = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'], yticklabels=['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[67]:


y_pred_test = model.predict(test_scaled)
y_pred_test =np.argmax(y_pred_test, axis=1)

y_test =np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'], yticklabels=['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[ ]:




