
import os

import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
from numpy import save,load
#

base_path = ('/home/maira/github/ResultsJournalSoftSensor/Soft-Sensing/')
result_directory = os.path.join(base_path,'Paper-R2-Plot')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
data_array_path = (os.path.join(base_path, '/home/maira/github/AmmoniumSensorApproximator/Data'))

# Load Numpy array of inputs and ground Truth

X_train = load(os.path.join(data_array_path,'X_train.npy'))
X_valid = load(os.path.join(data_array_path,'X_valid.npy'))
X_test = load(os.path.join(data_array_path,'X_test.npy'))
Y_test = load(os.path.join(data_array_path,'nh4_test.npy'))


best_GRU_CNN = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/GRUconv'))
best_LSTM = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/LSTM'))
best_GRU = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/GRU'))
best_BiGRU = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/BIGRU'))
best_BiLSTM = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/BILSTM'))
best_CNN = tf.keras.models.load_model(os.path.join('/home/maira/gitSubmission/AmmoniumSensorApproximator/TrainedModels/CNN'))


GRUconv_test_pred = best_GRU_CNN.predict(X_test).reshape(-1)
GRUconv_valid_pred = best_GRU_CNN.predict(X_valid).reshape(-1)
GRUconv_train_pred = best_GRU_CNN.predict(X_train).reshape(-1)

LSTM_test_pred = best_LSTM.predict(X_test).reshape(-1)
LSTM_valid_pred = best_LSTM.predict(X_valid).reshape(-1)
LSTM_train_pred = best_LSTM.predict(X_train).reshape(-1)

GRU_test_pred = best_GRU.predict(X_test).reshape(-1)
GRU_valid_pred = best_GRU.predict(X_valid).reshape(-1)
GRU_train_pred = best_GRU.predict(X_train).reshape(-1)

BiGRU_test_pred = best_BiGRU.predict(X_test).reshape(-1)
BiGRU_valid_pred = best_BiGRU.predict(X_valid).reshape(-1)
BiGRU_train_pred = best_BiGRU.predict(X_train).reshape(-1)

BiLSTM_test_pred = best_BiLSTM.predict(X_test).reshape(-1)
BiLSTM_valid_pred = best_BiLSTM.predict(X_valid).reshape(-1)
BiLSTM_train_pred = best_BiLSTM.predict(X_train).reshape(-1)


CNN_test_pred = best_CNN.predict(X_test).reshape(-1)
CNN_valid_pred = best_CNN.predict(X_valid).reshape(-1)
CNN_train_pred = best_CNN.predict(X_train).reshape(-1)



def r2(Y_test, Y_pred, legendTitle,row,col,plotnumber):
    plt.subplot(row,col,plotnumber)
    Y_test = Y_test.reshape(-1)
    r_squared = r2_score(Y_test, Y_pred)
    plt.scatter(Y_test, Y_pred, s=10)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')

    plt.plot(np.unique(Y_test), np.poly1d(np.polyfit(Y_test, Y_pred, 1))(np.unique(Y_test)))  # color = 'red'
    # plt.title( "R2: "+r_squared)
    plt.annotate("R-squared = {:.4f}".format(r_squared), (0.1, 0.6))

    plt.legend([legendTitle],loc='lower right')
    plt.grid(True)


    #plt.show()
def rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    return  rmse
plt.figure(figsize=(24,8))
r=3
c=2
r2(Y_test,GRUconv_test_pred,'GRUconv',r,c,1)
r2(Y_test,GRU_test_pred,'GRU',r,c,2)
r2(Y_test,BiGRU_test_pred,'Bi-GRU',r,c,3)
r2(Y_test,LSTM_test_pred,'LSTM',r,c,4)
r2(Y_test,BiLSTM_test_pred,'Bi-LSTM',r,c,5)
r2(Y_test,CNN_test_pred,'CNN',r,c,6)

plt.show()

print("-----------Root Mean Squared Error-----------")
print("GRUconv RMSE = {:.5f}".format(rmse(Y_test,GRUconv_test_pred)))
print("LSTM RMSE = {:.5f}".format(rmse(Y_test,LSTM_test_pred)))
print("GRU RMSE = {:.5f}".format(rmse(Y_test,GRU_test_pred)))
print("Bi-GRU RMSE = {:.5f}".format(rmse(Y_test,BiGRU_test_pred)))
print("Bi-LSTM RMSE = {:.5f}".format(rmse(Y_test,BiLSTM_test_pred)))
print("CNN RMSE = {:.5f}".format(rmse(Y_test,CNN_test_pred)))
print("-----------Root Mean Squared Error-----------")