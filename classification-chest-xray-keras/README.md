# **Chest_Xray using Deep Learning and Keras Library**
![photo](https://i.imgur.com/jZqpV51.png)

**This is an Educational Project for Graduates to practise the following tasks:**

*(I do not claim any innovation or outstanding performance)*

1. Image classification using deep learning and Keras library.
2. Loading grayscale images in JPEG forma, creating a dataset and generating corresponding lables.
3. Handling imbalanced dataset and improving model performance.

## Dataset Information
The dataset is available in the following link as *"ChestXRay2017.zip"*:
- (https://data.mendeley.com/datasets/rscbjbr9sj/2)


### Content of Images

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

**Note**: Delete the "**.DS_Store**" after unzipping.


 ## Mounting Drive on your Google Colab
 ```
from google.colab import drive
drive.mount('/content/drive')
```
## Import Required Libraries
```
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
```
## Defining Functions
```
def plot_figures(index, img_array1, img_array2):
  plt.subplot(2,3,1)
  plt.imshow(img_array1[index[0],:,:,:],cmap='gray')
  plt.subplot(2,3,2)
  plt.imshow(img_array1[index[1],:,:,:],cmap='gray')
  plt.subplot(2,3,3)
  plt.imshow(img_array1[index[2],:,:,:],cmap='gray')

  plt.subplot(2,3,4)
  plt.imshow(img_array2[index[0],:,:,:],cmap='gray')
  plt.subplot(2,3,5)
  plt.imshow(img_array2[index[1],:,:,:],cmap='gray')
  plt.subplot(2,3,6)
  plt.imshow(img_array2[index[2],:,:,:],cmap='gray')
```

```
def plot_hist(hist):
  plt.figure(figsize=(12, 6))
  plt.subplot(1,2,1)
  plt.plot(hist.history["loss"])
  plt.plot(hist.history["val_loss"])
  plt.title("model loss")
  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.legend(["train", "validation"], loc="upper left")
  plt.ylim(0, 10)

  plt.subplot(1,2,2)
  plt.plot(hist.history["accuracy"])
  plt.plot(hist.history["val_accuracy"])
  plt.title("model accuracy")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.legend(["train", "validation"], loc="upper left")
  plt.show()
```
## DATA LOADING
**We changed this code of Keras for loadind JPEG format of data.**

### Import Required Libraries
```
from keras.utils import load_img, img_to_array
from tensorflow import one_hot
from os import listdir
from os.path import join
```

### Define Function
```
def Load_Data(path, IMG_SIZE):  # IMG_SIZE = 224  # Based on our networks examples, original images have different sizes.
  Num_Classes = 2  # for one-hot
  folders = listdir(path)
  images = []
  labels= []
  ix_PNEUMONIA = 0; ix_NORMAL=0

  for folder in folders:  # folder = 'NORMAL'
    path2 = join( path, folder )
    files = listdir( path2 )

    for file in files: # file = 'Copy of IM-0115-0001.jpeg'
      path3 = join(path2, file )
      img = load_img(
          path3,
          color_mode = "rgb",
          target_size = (IMG_SIZE, IMG_SIZE), # None
          interpolation= "bilinear",
          keep_aspect_ratio=False,  # True: The image is cropped in the center with target aspect ratio before resizing.
          )
      img = img_to_array( img )     # Converts a PIL Image instance to a NumPy array
      # img = np.uint8(img)         # if you have a huge dataset, or your network input should be uint
      images.append( img )

      if folder == 'NORMAL':
        labels.append( 0 )
        ix_NORMAL+=1
      elif folder == 'PNEUMONIA':
        labels.append( 1 )
        ix_PNEUMONIA+=1
      else:
        raise ValueError('Check the folder name')
  images = np.array( images )
  labels = np.array( labels )
  # print( DataFrame(labels, columns=['Label']).to_string() )
  labels = one_hot(labels, Num_Classes)
  # print(DataFrame(labels).to_string())
  plt.imshow( np.uint8(img) )
  print('Normal Images: ', ix_NORMAL)
  print('PNEUMONIA Images: ', ix_PNEUMONIA)

  return images, labels
```
#### Train Data
```
train_images, train_labels = Load_Data(path, IMG_SIZE = 224)
                                      # 224 is based on our network input size and its examples on Keras.
```                                      
> The output will be a sample of your train dataset:

#### Validation Data
*If you have a separate validation data use this. But if not, you can use **`train_test_split`** for making validation dataset.*
```
valid_images, valid_labels = Load_Data(path, IMG_SIZE = 224)
```                                     
#### Test Data
```
test_images, test_labels = Load_Data(path, IMG_SIZE = 224)
```                                 

## Data Augmentation
**Note**: You can apply augmentation layers as preprocessing layers of the network. By default, augmentation layers are only applied during training.
In Keras 3 the number of Augmentation Layers has been expanded. 
For this project, we used three Augmentation layers 
- Keras 2 API (https://keras.io/2.15/api/layers/preprocessing_layers/image_augmentation)
- Keras 3 API (https://keras.io/api/layers/preprocessing_layers/image_augmentation)
```
from keras import layers
```
```
Rot_layer = layers.RandomRotation(factor=0.05, fill_mode="constant")
                                  # factor: a float represented as fraction of 2 Pi,    0.05 ≃ 20 degrees
                                  # fill_mode="constant": filling all values beyond the edge with k = 0.
Flip_layer = layers.RandomFlip(mode="horizontal")
Contrast_layer = layers.RandomContrast(factor=0.2)
                                  # The contrast_factor will be randomly picked between [1.0 - factor, 1.0 + factor]
                                  # For any pixel x in the channel, the output will be ((x - mean) * contrast_factor + mean) where mean is the mean value of the channel.
```
### Concatenate the created augmentation layers with train_images and their labels
```
train_images_aug = np.concatenate((train_images, Rot_layer(train_images), Contrast_layer(train_images)), axis=0)
train_labels_aug = np.concatenate((train_labels, train_labels, train_labels), axis=0)
```
## Callbacks 
- Available callbacks in Keras (https://keras.io/api/callbacks)

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.
You can pass a list of callbacks to the .fit() method of your model.
We used two Callbacks 
```
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
```
```
my_callback = [EarlyStopping(monitor='val_loss', patience = 10),
               ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)]
```
## Model
> We used Keras Models **`EfficientNetB0`** and **`InceptionV3`** with weights pre-trained on imagenet.

> **EfficientNet**
> - (https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning)
>
> Each Keras model expects a specific kind of input preprocessing.
> 
> For EfficientNet, input preprocessing is included as part of the model.
> - (https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input)
> 

### Check min and max values of our dataset
> EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
```
print(  'train images min max: ', train_images.min(), train_images.max(),
      '\nvalid images min max: ', valid_images.min(), valid_images.max(),
      '\ntrain images min max: ', test_images.min(), test_images.max() )
```
### Define Model
```
from keras.applications import EfficientNetB0
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
```
```
Num_Classes = 2  #  We have two classes.
IMG_SIZE = 224   # 224 is based on our network (EfficientNetB0) input size.

# Pre-trained model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = EfficientNetB0( weights='imagenet', input_tensor=inputs, include_top=False )
                             # This model takes input images of shape (224, 224, 3), and the input data should be in the range [0, 255].
                             # Normalization is included as part of the model.
                             # Include_top is set to False, as we want to rebuild the top layaers of the network.

# Freeze the pretrained weights
base_model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D()(base_model.output) # GlobalAveragePooling2D: Average of feature maps
x = layers.BatchNormalization()(x)
x = layers.Dropout( rate = 0.2 )(x)
                          # sets input units to 0 with a frequency of rate at each step
                          # Inputs not set to 0 are scaled up by 1 / (1 - rate) such that the sum over all inputs is unchanged.
outputs = layers.Dense(Num_Classes, activation="softmax")(x)

model = Model(inputs = base_model.input, outputs = outputs)
# model.summary()

# Compile Model
optimizer = Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
```

## Train the Model
```
epochs = 50
batch_size = 4
hist = model.fit(
                 x = train_images,  # train_images_aug,
                 y = train_labels,  # train_labels_aug,
                 batch_size = batch_size,
                 epochs = epochs,
                 # callbacks= my_callback,
                 # validation_split=0.2,  # use this if you don't have validation data 
                 validation_data=(valid_images, valid_labels),
                 shuffle=True,
                 )

# Save Model
model.save('/content/drive/MyDrive/Projects/Classification_Chest_Xray_Keras/saved_Models/EfficientNet.keras')
```

**InceptionV3**
> - (https://keras.io/api/applications/)
>
> Call keras.applications.inception_v3.preprocess_input on your inputs before passing them to the model.
> - (https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input)
>   
> Input: floating point numpy.array or a backend-native tensor, 3D or 4D with 3 color channels, with values in the range [0, 255].
> 
> The preprocessed data are written over the input data if the data types are compatible. To avoid this behaviour, numpy.copy(x) can be used.
                                                                                          train_images2 = preprocess_input( np.copy(train_images) )

### Required Libraries
```
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
```

### Data Pre-processing (values are scaled between -1 and 1, type float 32)
```
train_images = preprocess_input(train_images)
# train_images2 = preprocess_input( np.copy(train_images) )
valid_images = preprocess_input(valid_images)
# valid_images2 = preprocess_input( np.copy(valid_images) )
```
### Pre-trained model
```
base_model = InceptionV3(weights='imagenet', include_top=False)
base_model.trainable = False
```

### Rebuild top
```
Num_Classes = 2

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = Dense(Num_Classes, activation='softmax')(x)

model2 = Model(inputs=base_model.input, outputs=outputs) # model.summary()

# Compile Model
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
```

## Train the model
```
epochs = 50
batch_size = 4
hist2 = model2.fit(
                 x = train_images,  # train_images_aug,
                 y = train_labels,  # train_labels_aug,
                 batch_size = batch_size,
                 epochs = epochs,
                 # callbacks= my_callback,
                 # validation_split=0.2,  # use this if you don't have validation data 
                 validation_data=(valid_images, valid_labels),
                 shuffle=True,
                 )
# Save Model
model2.save('/content/drive/MyDrive/Projects/Classification_Chest_Xray_Keras/saved_Models/InceptionV3.keras')
```

