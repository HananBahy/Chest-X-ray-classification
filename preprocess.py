import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


### --------------- Preprocessing ------------
images = glob.glob("images/*")
print(images[0])
print(len(images))
# img_size = (224, 224)

for img in images:
    img2 = cv2.imread(img)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    r = 224.0 / img2.shape[0]
    dim = (int(img2.shape[1] * r), 224)
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    # np.save("preprocessed/"+img.split(".")[0]+".npy", resized)
    cv2.imwrite("preprocessed/"+img.split(".")[0]+".jpg", resized)

