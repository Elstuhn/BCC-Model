from keras.models import load_model
from tensorflow.keras.datasets import mnist
from time import sleep
from matplotlib import image
import matplotlib.pyplot as plt
import pickle
model = load_model('BloodCells.hdf5')
loadables = ["trainX", "testX", "trainy", "testy"]
classes = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
for i in loadables:
    with open(f"Blood Cells/{i}", "rb") as readfile:
        exec(f"{i} = pickle.load(readfile)")

predicted = testX[987]
answer = classes[testy[987]]
plt.figure(figsize=(10,10))
plt.title("Picture Input")
plt.imshow(predicted)
plt.show()
prediction = model.predict(predicted.reshape(1, 120, 160, 3))
print(f"Prediction: {classes[prediction.argmax()]}")
print(f"Actual: {answer}")

def actual_predictions(prediction):
    fig = plt.figure()
    plt.ylabel("Percentage")
    plt.xlabel("Blood Cell Types")
    prediction = prediction[0]
    plt.bar(classes, [round(i*100, 1) for i in prediction])
    plt.show()

actual_predictions(prediction)
    

def check_acc(model, X_test, y_test):
    incorrect = 0
    correct = 0
    for i in range(len(X_test)):
        prediction = model.predict(X_test[i].reshape(1, 120, 160, 3)).argmax()
        actual = y_test[i]
        if int(prediction) != int(actual):
            incorrect += 1
        else:
            correct += 1
            
    fig = plt.figure()
    plt.ylabel("Amount")
    plt.xlabel(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    plt.bar(["correct", "incorrect"], [correct, incorrect])
    plt.show()
    print(f"Accuracy: {(correct/(correct+incorrect))*100}%")

check_acc(model, testX, testy)
