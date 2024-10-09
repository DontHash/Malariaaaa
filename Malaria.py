import numpy as np 
import pandas as pd 
import os
import tensorflow as tf 
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense, Flatten,Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import cv2
from tensorflow.keras.optimizers import Adam

data_path = '/Dataset_Path_Here/'
(os.listdir('/Dataset_Path_Here/'))
y=[]
for i in range(0,2):
    y.append(os.listdir('/Dataset_Path_Here/')[i])
print(y)


#Calculating the mean Dimension to normalize the Images
dimension_1 = []
dimension_2 = []
base_path = 'Image_Dataset_directoryParasitized/'


for image_name in os.listdir(base_path):
    img_path = os.path.join(base_path, image_name)
    
   
    img = cv2.imread(img_path)
    
    if img is not None:
        # Get the dimensions of the image
        d1, d2, d3 = img.shape
        dimension_1.append(d1)
        dimension_2.append(d2)
    else:
        print(f"Error loading image: {img_path}")

d1 = np.mean(dimension_1)
d2 = np.mean(dimension_2)
img_shape = (134,134,3)

#Checking if you need to rescale the data or not
base_path = 'Any_Folder_that_contains_an_Image'
cell = cv2.imread(os.path.join(base_path,'C99P60ThinF_IMG_20150918_141001_cell_99.png'))#This stores the image in cell
cell.max()

#ImageData Generator in which You can set the parameters
imageGen = ImageDataGenerator(rotation_range = 15,
                              width_shift_range = 0.08,
                              height_shift_range = 0.08,
                              shear_range = 0.08,
                              zoom_range = 0.1,
                              horizontal_flip = True,
                              fill_mode = 'nearest',
                              rescale = 1/255
)

#The Image Data Generator automatically Identifies classes, Since there was a third Folder named cell_images, That was to be excluded as that was not an image class
imageGen.flow_from_directory('Image_Dataset_directory')
print(os.listdir("Image_Dataset_directory"))
class_indices = {'Uninfected': 0, 'Parasitized': 1, 'cell_images': 2}  # Adjust these names as needed

# Step 2: Create a custom function to filter out the 3rd class
def filter_classes(label):
    return label if label < 2 else None 

train_generator = imageGen.flow_from_directory(
    'Image_Dataset_directory',
    target_size=(134, 134),
    color_mode = 'rgb',
    batch_size=32,
    shuffle=True,
    classes=list(class_indices.keys())[:2],  # Only include the first two classes(Uninfected and parasitized)
    class_mode='binary'
)

#Data splitting into Train test and Validation
data, labels = [], []
for i in range(len(train_generator)):
    batch = next(train_generator)
    data.extend(batch[0])
    labels.extend(batch[1])
    if len(data) >= train_generator.samples:
        break

data = np.array(data)
labels = np.array(labels)

X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of test samples: {len(X_test)}")

train_generator = imageGen.flow(X_train, y_train, batch_size=32)
val_generator = ImageDataGenerator(rescale=1/255).flow(X_val, y_val, batch_size=32)
test_generator = ImageDataGenerator(rescale=1/255).flow(X_test, y_test, batch_size=32)

#Custom CNN architechture to train the model 

model = Sequential([
    # First Convolutional Block
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(134,134,3), padding='same'),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#EarlyStopping and ModelCheckpoint to ensure best Epoch gets saved
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss',patience=0,restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'test_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32,
    epochs=50,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(test_generator, verbose=2)

print("Training Finished")
#Saving the Model
model.save('final_model_1.keras')
