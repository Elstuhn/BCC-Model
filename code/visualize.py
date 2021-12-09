import matplotlib.pyplot as plt
import numpy as np

def check_acc(model, X_test, y_test):
    incorrect = 0
    correct = 0
    for i in range(len(X_test)):
        prediction = model.predict(X_test[i].reshape(1, 120, 160, 1)).argmax()
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

    
def plothist(history, metric):
    plt.figure()
    if metric == "accuracy":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel("Accuracy")
    elif metric == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
