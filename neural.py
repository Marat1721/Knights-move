import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import layers
from keras import models

import tensorflow as tf
import cv2                  
import numpy as np  
from tqdm import tqdm
import os

X=[]
Z=[]
IMG_SIZE=75
BLANK_DIR= r'figures\blank'
WBISHOP_DIR= r'figures\white\b'
WKING_DIR= r'figures\white\k'
WQUEEN_DIR= r'figures\white\q'
WKNIGHT_DIR= r'figures\white\n'
WPAWN_DIR= r'figures\white\p'
WROOK_DIR= r'figures\white\r'
BBISHOP_DIR= r'figures\black\b'
BKING_DIR= r'figures\black\k'
BQUEEN_DIR= r'figures\black\q'
BKNIGHT_DIR= r'figures\black\n'
BPAWN_DIR= r'figures\black\p'
BROOK_DIR= r'figures\black\r'

def make_train_data(figure_type,DIR):
    for img in tqdm(os.listdir(DIR)[:290]):
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        Z.append(str(figure_type))

make_train_data('o', BLANK_DIR)
make_train_data('wb', WBISHOP_DIR)
make_train_data('wk', WKING_DIR)
make_train_data('wq', WQUEEN_DIR)
make_train_data('wn', WKNIGHT_DIR)
make_train_data('wp', WPAWN_DIR)
make_train_data('wr', WROOK_DIR)
make_train_data('bb', BBISHOP_DIR)
make_train_data('bk', BKING_DIR)
make_train_data('bq', BQUEEN_DIR)
make_train_data('bn', BKNIGHT_DIR)
make_train_data('bp', BPAWN_DIR)
make_train_data('br', BROOK_DIR)

print(len(X))

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = tf.keras.utils.to_categorical(Y, 13)
X = np.array(X)
X = X/255.
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
      rotation_range=40,
      shear_range=0.2,
      zoom_range=0.2,
      vertical_flip=True,
      horizontal_flip=True)

train_datagen.fit(x_train)

base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation ='relu')(x)
predictions = layers.Dense(13, activation ='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

adam = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

History = model.fit_generator(train_datagen.flow(x_train,y_train, batch_size=290),
                              epochs = 15, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // 290)

model.save('model.h5')