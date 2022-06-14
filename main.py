import pandas as pd
import plotly.graph_objs as go
import sklearn.metrics
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping



def generateArrayWith0And1(length):
    array = []
    for i in range(length):
        array.append(random.randint(0, 1))
    return array


def computeSuccessFromConfusionMatrix(matrix):
    allSum = 0
    for i in matrix:
        for a in i:
            allSum = allSum + a

    diagonal = matrix[0][1] + matrix[1][0]

    return (diagonal / allSum) * 100


def convertNpArrayToIntArray(npArray):
    arr = []
    for a in npArray:
        arr.append(int(a[0]))
    return arr


# Nastavil som aby sa zobrazovali všetky stlpce vo výpise
pd.set_option('display.expand_frame_repr', False)

# Načítavam dáta
testData = pd.read_csv("water_test.csv")
trainData = pd.read_csv("water_train.csv")

# Na prázdne miesta tam kde nie je žiadna hodnota doplňujem 0
testData = testData.fillna(0)
trainData = trainData.fillna(0)

testDataWithoutPotability = testData.drop(['Potability'], axis=1)
trainDataWithoutPotability = trainData.drop(['Potability'], axis=1)

# Normalizujem data
scaler = MinMaxScaler()
testDataNormalized = pd.DataFrame(scaler.fit_transform(testDataWithoutPotability),columns=testDataWithoutPotability.columns)
trainDataNormalized = pd.DataFrame(scaler.fit_transform(trainDataWithoutPotability),columns=trainDataWithoutPotability.columns)

#
# Priemery a odchýlky
#

print("------------------------------------------------------------------------")
print("Primery a štandardné odchylky")
print("------------------------------------------------------------------------")

# Vypisujem premer, a štandardné odchýlky
print("Trenovacie data pred normalizáciou - priemer, štandardná odchýlka")
print(trainData.describe())
print("Trenovacie data po normalizácií - priemer, štandardná odchýlka")
print(trainDataNormalized.describe())

print("--------------------------------------------------------------------")

print("Testovacie data pred normalizáciou - - priemer, štandardná odchýlka")
print(testData.describe())
print("Testovacie data po normalizácií - priemer, štandardná odchýlka")
print(testDataNormalized.describe())

#
# Histogramy
#

print("------------------------------------------------------------------------")
print("Histogramy")
print("------------------------------------------------------------------------")

# Histogram - Trenovacie data pred normalizáciou - stlpec ph
plt.hist(trainData["ph"])
plt.title("Histogram - Trenovacie data pred normalizáciou - stlpec ph")
plt.show()

# Histogram - Trenovacie data po normalizácií - stlpec ph
plt.hist(trainDataNormalized["ph"])
plt.title("Histogram - Trenovacie data po normalizácií - stlpec ph")
plt.show()

print("-------------------------------------------------")
# Histogram - Testovacie data pred normalizáciou - stlpec ph
plt.hist(testData["ph"])
plt.title("Histogram - Testovacie data pred normalizáciou - stlpec ph")
plt.show()

# Histogram - Testovacie data po normalizácií - stlpec ph
plt.hist(testDataNormalized["ph"])
plt.title("Histogram - Testovacie data po normalizácií - stlpec ph")
plt.show()

#
# Uspešnosť pri náhodnom klasifikátore
#

print("------------------------------------------------------------------------")
print("Uspešnosť pri náhodnom klasifikátore")
print("------------------------------------------------------------------------")

truePotabilities = testData["Potability"].array
randomPotabilities = generateArrayWith0And1(len(truePotabilities))
matrix = sklearn.metrics.confusion_matrix(truePotabilities, randomPotabilities)
print(matrix)
print("Úspešnsoť pri náhodnom klasifikátore:")
print(computeSuccessFromConfusionMatrix(matrix))

#
# logisticka regresia
#

print("------------------------------------------------------------------------")
print("logisticka regresia")
print("------------------------------------------------------------------------")

potability = trainData.Potability
potabilityTest = testData.Potability
regression = LogisticRegression().fit(trainDataNormalized, potability.values.ravel())
regresionResult = regression.predict(testDataNormalized)
print(classification_report(potabilityTest, regresionResult, zero_division=0))

testDataLabels = testData.iloc[:, 9:].values
trainDataLabels = trainData.iloc[:, 9:].values

testDataLabels = np.array(testDataLabels.astype(np.float))
trainDataLabels = np.array(trainDataLabels.astype(np.float))

testDataNormalizedX = np.array(testDataNormalized)
trainDataNormalizedX = np.array(trainDataNormalized)


trainDataNormalizedX, validDataX, trainDataLabels, validDataLabels = train_test_split(trainDataNormalizedX, trainDataLabels, test_size=0.20, random_state=42)

print("dlzky")
print(len(trainDataNormalizedX))
print(len(validDataX))


earlyStopping = EarlyStopping(monitor='val_loss', patience=23)

neuralNetsModel = Sequential()
neuralNetsModel.add(Dense(62, activation='relu'))
neuralNetsModel.add(Dense(62, activation='relu'))
neuralNetsModel.add(Dense(62, activation='relu'))
neuralNetsModel.add(Dense(1, activation='sigmoid'))


optimizer = Adam(learning_rate=0.001)
optimizer1 = Adadelta(learning_rate=0.001)
optimizer2 = Adagrad(learning_rate=0.00001)
neuralNetsModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
trainedNet = neuralNetsModel.fit(trainDataNormalizedX, trainDataLabels, epochs=1000, verbose=0,
                                 validation_data=(validDataX, validDataLabels), callbacks=[earlyStopping])


trainingHistory = pd.DataFrame.from_dict(trainedNet.history)

lineChartsOfTraining = make_subplots(rows=1, cols=2, subplot_titles=['Accuracy', 'Loss'])
lineChartsOfTraining.add_trace(go.Scatter(y=trainingHistory['loss'], name='loss', mode='lines'), row=1, col=2)

lineChartsOfTraining.add_trace(go.Scatter(y=trainingHistory['val_loss'], name='val_loss', mode='lines'), row=1, col=2)

lineChartsOfTraining.add_trace(go.Scatter(y=trainingHistory['accuracy'], name='accuracy', mode='lines'), row=1, col=1)

lineChartsOfTraining.add_trace(go.Scatter(y=trainingHistory['val_accuracy'], name='val_accuracy', mode='lines'), row=1, col=1)

lineChartsOfTraining.show()


outputPredictions = neuralNetsModel.predict(testDataNormalizedX)

outputPredictions = np.round(outputPredictions)


print(classification_report(testDataLabels, outputPredictions))

confusion = pd.DataFrame(confusion_matrix(testDataLabels, outputPredictions))
confusionMatrixPlot = go.Figure(data=go.Heatmap(z=confusion, x=['nepitna', 'pitna'], y=['nepitna', 'pitna'], colorscale='blues'))


confusionMatrixPlot.show()
