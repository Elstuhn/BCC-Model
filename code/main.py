from model import *
from data-clean import *
from visualize import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

trainX = data-clean.trainX
testX = data-clean.testX
trainy = data-clean.labels
testy = data-clean.labelsY

earlystop = EarlyStopping(monitor = "val_accuracy", patience = 3)
model_save = ModelCheckpoint('BloodCells.hdf5',
                            save_best_only=True)

model = makemodel()
history = model.fit(trainX, trainy, epochs=1000, validation_data = (testX, testy), callbacks = [earlystop, model_save], batch_size = 125)
score = model.evaluate(testX, testy, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

check_acc(model, testX, testy)
    
plothist(history, "accuracy")
plothist(history, "loss")
