import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


### --------------- Preprocessing ------------
# images = glob.glob("images/*")
# print(images[0])
# print(len(images))
# img_size = (224, 224)

# for img in images:
#     image = cv2.resize(cv2.imread(img) ,img_size).astype(np.float32)
#     print(image)
#     # print(img.split(".")[0])
#     np.save("preprocessed/"+img.split(".")[0]+".npy", image)


imagesPaths = glob.glob("preprocessed/images/*")
# print(imagesPaths[0])
# print(len(imagesPaths))

### indecis of images
imagesIndecis = [p.split("\\")[1].split(".")[0] for p in imagesPaths]
# print(imagesIndecis[0])

data = pd.read_csv("Data_Entry_2017.csv")
data = data.rename(columns={'Image Index': 'Idx', 'Finding Labels': 'class'})
data = data.iloc[:, :2]
# print(data.head(1))

labelDict = dict()
for i in data.values:
    index = i[0].split(".")[0]
    labelDict[index] = i[1]
# print(labelDict["00000001_000"])

for k,v in labelDict.items():
    if v == "No Finding":
        v = 'NoFinding'
    labelDict[k] = ' '.join(v.split("|"))

labels = [labelDict[ind] for ind in imagesIndecis]
labelsVectorized = []
# get Vectors for all labels
vectorizer = CountVectorizer()
vectorizer.fit(labels)

for label in labels:
    if label == "No Finding":
        label = 'NoFinding'
    labelsVectorized.append(vectorizer.transform([label]).toarray()[0])

# print(labelsVectorized[:10])

X_train_paths, X_test_paths, y_train, y_test = train_test_split(imagesPaths, labelsVectorized, test_size = 0.15, random_state = 0)

## ---------------- Data Generator -----------------

def dataGenerator(imgs_paths, mylabels, batchSize):
    i = 0
    # file_list = os.listdir(directory)
    while True:
        image_batch = []
        labels_batch = []
        for b in range(batchSize):
            if i == len(imgs_paths):
                i = 0
                # random.shuffle(file_list)
            img_sample = cv2.imread(imgs_paths[i])
            label_sample = mylabels[i]
            i += 1
            # image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
            image_batch.append(img_sample)
            labels_batch.append(label_sample)

        yield ([np.array(image_batch), np.array(labels_batch)])

### ------------------- MODEL NOW ---------------------

from keras.models import Model, Input
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_uniform
from resnet152 import ResNet152
from keras import backend as K
from metrics import f_score
import f1score

num_classes = 15

########### FIRST TRAINED MODEL #############
# model = ResNet152(weights='imagenet')
# model.layers.pop()
# for layer in model.layers:
#     layer.trainable=True
# last = model.layers[-1].output
# x = Dense(num_classes, activation='sigmoid', name='fc' + str(num_classes), kernel_initializer = glorot_uniform(seed=0))(last)
# model = Model(inputs = model.input, outputs = x, name='ResNet50')
#############################################

base_network = ResNet152(include_top=False, weights='imagenet')
for layer in base_network.layers:
    layer.trainable=True
img_input = Input(shape=(224, 224, 3))
model = base_network(img_input)
# output layer
model = Flatten()(model)
model = Dense(num_classes, activation='sigmoid', name='fc' + str(num_classes), kernel_initializer = glorot_uniform(seed=0))(model)
# Create model
final_model = Model(inputs = img_input, outputs = model, name='ResNet152')

learning_rate=.001     #learning rate
# opt = Adam()  #Adam optimizer
opt = SGD(lr=learning_rate ,momentum=.9 ,decay=0.0005)
final_model.compile(optimizer=opt,loss='binary_crossentropy',metrics=[f1score.f1])

### ----------------- Load Latest Weights --------------------
final_model.load_weights("model/weights.31-0.08.hdf5", by_name=True)

from keras.callbacks import ModelCheckpoint ,TensorBoard
mc = ModelCheckpoint(              #for checkpoint
    filepath='model/weights.{epoch:02d}-{loss:.2f}.hdf5',
    monitor='loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='min',
    period=1)

tb=TensorBoard(           #to show results (loss ,accuracy) on tensorboard
    log_dir='./logs',
    histogram_freq=0,
    write_graph=True,
    write_images=False)

# callbacks = [mc,tb]

# imagesPaths = imagesPaths[:3000]
# labelsVectorized = labelsVectorized[:3000]
batch_size = 24
NUM_IMGS = len(imagesPaths)
print(len(imagesPaths))

final_model.fit_generator(
    dataGenerator(imagesPaths, labelsVectorized, batch_size),
    epochs=1000, 
    steps_per_epoch=int(NUM_IMGS/batch_size), 
    callbacks=[mc, tb],
    initial_epoch=31)

# IMAGES = np.array([cv2.imread(im) for im in imagesPaths])
# LABELS = np.array(labelsVectorized)
# final_model.fit(x=IMAGES, y=LABELS, batch_size=batch_size, epochs=100, callbacks=[mc, tb])
