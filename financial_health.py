import gc
import os
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')
#####################################
####### load dataset
PATH='D:/customers_financial_health_to_achieve_their_monetary_goals/'
os.listdir(PATH)
df_train= pd.read_csv(PATH+'train.csv')
df_test= pd.read_csv(PATH+'test.csv')
#####################################
df_train.columns
df_train.info()
df_train.describe()
df_train.dtypes
df_test.dtypes
df_train.head()

# we have in df_train >> ID_code(string), target(output), features(var_0:var_199)
# we have in df_test >> ID_code(string), features(var_0:var_199)
df_train.shape
train= df_train.drop(['ID_code'],axis=1)
train.shape

test= df_test.drop(['ID_code'],axis=1)
test.shape
#################################### missig values #############
missing_train= train.isnull().sum()
missing_test= test.isnull().sum()

################################### Features correlation ######################
# correlation matrix between each two variables to take quick overview of its relationships
#correlations = train.corr(method='pearson').abs().unstack().sort_values(kind="quicksort").reset_index()
##### correlations between features and target
correlations = train.corr()
correlations.head()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(correlations, vmax=.8, square=True)

k=57
correlations_wit_target = correlations.nlargest(k, 'target')['target'].index # pick the best powerfull Correlation(more 30%)
cm = np.corrcoef(train[correlations_wit_target].values.T)
f, ax = plt.subplots(figsize=(20, 15))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, yticklabels=correlations_wit_target.values, xticklabels=correlations_wit_target.values)
plt.show()
train= train[correlations_wit_target]

a= correlations_wit_target.drop(['target'])
test= test[a]
############################################ scaling ###############
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
scaled = minmax.fit_transform(train)
train= scaled

scaled = minmax.fit_transform(test)
test= scaled
############################# splite the train data ##########################################

from sklearn.model_selection import train_test_split
train=pd.DataFrame(train)
x=train.drop([0], axis=1)
y=train[0]
y = y.values.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()
y.shape # (200000, 2)
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=128)
X_train.shape
y_train.shape
y_valid.shape

####################### Neural Network using Tensorflow and Keras ##################
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(128, input_dim=56, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
############## save model 

from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("D:/customers_financial_health_to_achieve_their_monetary_goals/NN.json", "w") as json_file:
    json_file.write(model_json)
#####################

es = EarlyStopping(monitor='val_loss', patience=8)
mc = ModelCheckpoint('D:/customers_financial_health_to_achieve_their_monetary_goals/NN.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.1, epochs=40, batch_size=32, verbose=1,callbacks=[es,mc])
########### summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
########### summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
################################## convusion matrix ##################################
y_predict = model.predict(X_valid)

prediction = []
valid = []

for i in range(len(y_valid)): 
    prediction.append(np.argmax(y_predict[i]))
    valid.append(np.argmax(y_valid[i]))
### compartion
prediction = pd.Series(prediction)
valid = pd.Series(valid)
compare = pd.concat([prediction, valid], axis=1, keys=['prediction_valid', 'actual_valid'])
################# confusion_matrix ########################
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(prediction,valid)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
classes =[0,1]
plot_confusion_matrix(cm, classes)
#############################################################################
######## metrics ####################
from sklearn.metrics import classification_report
print ('Report : ')
print (classification_report(valid, prediction,digits=3 ))
##########################
####################################### predction test ######################
from keras.models import model_from_json
# load json and create model
json_file = open('D:/customers_financial_health_to_achieve_their_monetary_goals/NN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("D:/customers_financial_health_to_achieve_their_monetary_goals/NN.hdf5")
print("Loaded model from disk")
classifier.summary()
##############################
unseen_data_submission = classifier.predict_classes(test)
ID_code= df_test['ID_code']
unseen_data_submission = pd.DataFrame(unseen_data_submission)
ID_code = pd.Series(ID_code)
predict_submission = pd.concat([ID_code, unseen_data_submission], axis=1, keys=['actual_test', 'prediction_test'])


# save to csv file
predict_submission.to_csv('D:/customers_financial_health_to_achieve_their_monetary_goals/predict_submission.csv')

 