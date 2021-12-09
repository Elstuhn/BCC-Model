classes = ["Eosinophil", "Lymphocite", "Monocyte", "Neutrophil"]
labels = [0 for i in range(2497)] + [1 for i in range(2483)] + [2 for i in range(2478)] + [3 for i in range(2499)]
labelsY = [0 for i in range(623)] + [1 for i in range(620)] + [2 for i in range(620)] + [3 for i in range(624)]
labels = np.array(labels)
labelsY = np.array(labelsY)
trainX = []
testX = []
for files in os.listdir("TRAIN"):
    for image in os.listdir(f"TRAIN/{files}"):
        images = Image.open(f"TRAIN/{files}/{image}")
        images = images.resize((160, 120))
        images = np.array(images)
        trainX.append(images)
trainX = np.array(trainX)
trainX = trainX / 255.0

for files in os.listdir("TEST"):
    for image in os.listdir(f"TEST/{files}"):
        images = Image.open(f"TEST/{files}/{image}")
        images = images.resize((160, 120))
        images = np.array(images)
        testX.append(images)
testX = np.array(testX)
testX = testX / 255.0
