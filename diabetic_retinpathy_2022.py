from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from tqdm import tqdm_notebook as tqdm

from keras import layers
from keras.applications import DenseNet121, ResNet50V2
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score, accuracy_score

from google.colab import drive
drive.mount('/content/drive/')

path = '/content/drive/MyDrive/diabetic/'

df = pd.read_csv(path + 'train.csv')

df.head()

df['diagnosis'].value_counts().plot(kind = 'bar')

# Get Train images
files = os.listdir(path + 'train_images')

img_list = []

for i in files[:10]:
  img = cv2.imread(path + 'train_images/' + i)
  img = cv2.resize(src = img, dsize = (500, 500))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_list.append(img)

len(img_list)

plt.imshow(img_list[3])

img_list[3].shape

"""# **PRE-PROCESSING**"""

img = cv2.cvtColor(img_list[3], cv2.COLOR_RGB2GRAY)
plt.imshow(img, cmap = 'gray')

blur = cv2.GaussianBlur(src = img, ksize =(5,5), sigmaX = 0)
plt.imshow(blur, cmap = 'gray')

thres = cv2.threshold(blur,10 ,255, cv2.THRESH_BINARY)[1]
plt.imshow(thres, cmap='gray')

contour = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
contour

contour.shape

contour = contour[:,0,:]
contour

contour.shape

contour[:,0]

contour[:,0].argmax()

contour[356]

contour[:,0].argmin()

contour[120]

contour[:,1]

left = tuple(contour[contour[:,0].argmin()])
right = tuple(contour[contour[:,0].argmax()])
top = tuple(contour[contour[:,1].argmin()])
bottom = tuple(contour[contour[:,1].argmax()])

print(f'left:{left}, right:{right}, top:{top}, botton:{bottom}')

x1 = left[0]
x2 = right[0]
y1 = top[1]
y2 = bottom[1]

orj_img = img_list[3].copy()
plt.imshow(orj_img)

crop_img = orj_img[y1:y2, x1:x2]
plt.imshow(crop_img)

crop_img.shape

crop_img = cv2.resize(crop_img , (500, 500))
plt.imshow(crop_img)

crop_img.shape

"""**CLAHE**"""

lab = cv2.cvtColor(crop_img, cv2.COLOR_RGB2LAB)
lab.shape

l, a, b = cv2.split(lab)

plt.imshow(l, cmap= 'gray')
print("shape: ", l.shape)

"""**Flatten for histogram**"""

flatten = l.flatten()
print(flatten.shape)

plt.hist(flatten, 25, [0, 256], color = 'r')
plt.show()

CLAHE = cv2.createCLAHE(clipLimit = 7.0, tileGridSize = ((8,8)))
cl = CLAHE.apply(l)

plt.title('BEFORE CLAHE')
plt.imshow(l)

plt.title('AFTER CLAHE')
plt.imshow(cl)

"""**MERGE OTHER CHANNELS**"""

merge = cv2.merge((cl, a, b))

plt.imshow(merge)

final_img = cv2.cvtColor(merge, cv2.COLOR_LAB2RGB)
plt.imshow(final_img)

"""**Adding Median Blur to remove noise**"""

median_blur = cv2.medianBlur(final_img, ksize = 3)
plt.imshow(median_blur)

background = cv2.medianBlur(final_img, ksize = 49)
plt.imshow(background)

mask = cv2.addWeighted(median_blur, 1, background, -1, 255)
plt.imshow(mask)

plt.title('Before mask')
plt.imshow(median_blur)

final_mask = cv2.bitwise_and(mask, median_blur)
plt.title('After mask')
plt.imshow(final_mask)

from tqdm import tqdm_notebook as tqdm

files = os.listdir(path + "train_images")


def preprocessing(files):
    """
    This function returns images prepared for training
    """
    img_list = []
    for i in tqdm(files):
        image = cv2.imread(path + 'train_images/' + i)
        image = cv2.resize(image, (240, 240))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        copy = image.copy()
        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(copy, (5, 5), 0)

        thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)[1]

        # CONTOUR DETECTION
        contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0][0]
        contour = contour[:, 0, :]

        # GET COORDINATES
        x1 = tuple(contour[contour[:, 0].argmin()])[0]
        y1 = tuple(contour[contour[:, 1].argmin()])[1]
        x2 = tuple(contour[contour[:, 0].argmax()])[0]
        y2 = tuple(contour[contour[:, 1].argmax()])[1]

        #Crop Images Again to Destroy Black Area
        x = int(x2 - x1) * 4 // 50
        y = int(y2 - y1) * 5 // 50

        # THRES FOR CROPPED IMAGES
        copy2 = image.copy()
        if x2 - x1 > 100 and y2 - y1 > 100:
            copy2 = copy2[y1 + y: y2 - y, x1 + x: x2 - x]
            copy2 = cv2.resize(copy2, (240, 240))

        # LAB
        lab = cv2.cvtColor(copy2, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE - Contrast-Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=((8, 8)))
        cl = clahe.apply(l)

        # MERGING LAB
        merge = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(merge, cv2.COLOR_LAB2RGB)
        med_blur = cv2.medianBlur(final_img, 3)
        back_gorund = cv2.medianBlur(final_img, 37)

        # MASK FOR BLEEDING VEIN
        mask = cv2.addWeighted(med_blur, 1, back_gorund, -1, 255)
        final = cv2.bitwise_and(mask, med_blur)
        img_list.append(final)

    return img_list


img_list = preprocessing(files=files)

fig = plt.figure(figsize=(20,12))

for i in range(12):
  img = img_list[i]
  fig.add_subplot(3,4,i+1)
  plt.imshow(img)

plt.tight_layout()

"""**One hot encoding**"""

df['diagnosis']

y_train = pd.get_dummies(df['diagnosis']).values
y_train

"""**Ordinal Regression Encoding**"""

y_train_final = np.ones(y_train.shape, dtype = 'uint8')
y_train_final

y_train_final[:,4]= y_train[:,4]

y_train_final

for i in range(3, -1 , -1):
  y_train_final[:,i] = np.logical_or(y_train[:,i], y_train_final[:,i+1])
y_train_final

"""**Img to Array**"""

x_train = np.array(img_list)
x_train.shape

y_train_final.shape

"""# **Train Test split**"""

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42, shuffle = True)

print(f'x_train:{x_train.shape}',
      f'x_val:{x_val.shape}',
      f'y_train:{y_train.shape}',
      f'y_val:{y_val.shape}')

"""# **Data Generator**"""

datagen = ImageDataGenerator(horizontal_flip=True,
                             rescale = 1./255,
                             vertical_flip=True,
                             zoom_range=0.3,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode='constant',
                             cval=0.1)
val_datagen = ImageDataGenerator(
                             rescale = 1./255
                             )

batch_size = 32

data_generator = datagen.flow(x_train,
                              y_train,
                              batch_size=batch_size,
                              seed=42)
val_data_generator = val_datagen.flow(x_val,
                                      y_val,
                                      batch_size = batch_size,
                                      seed = 42)

"""# **Transfer Learning**"""

def build_model(base):
  model = Sequential()
  model.add(base)
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(5, activation = 'sigmoid'))

  model.compile(
      loss = 'binary_crossentropy',
      optimizer = Adam(),
      metrics = ['accuracy']
  )
  return model

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1

        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return

"""**DenseNet121**"""

base_DenseNet121 = ResNet50V2(
  weights='imagenet',
  include_top=False,
  input_shape=(240,240,3)
)

model_DenseNet121 = build_model(base = base_DenseNet121)
model_DenseNet121.summary()

lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    patience = 3,
    verbose = 1,
    mode = 'auto',
    min_lr = 0.000001
);

kappa = Metrics()

history_DenseNet121 =  model_DenseNet121.fit(data_generator,
                              steps_per_epoch = data_generator.n//batch_size,
                              epochs = 5,
                              validation_data = (x_val,y_val),
                              validation_steps= val_data_generator.n//batch_size,
                              callbacks = [lr])

