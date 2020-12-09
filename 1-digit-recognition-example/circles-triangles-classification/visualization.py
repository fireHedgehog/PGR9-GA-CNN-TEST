import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras import Sequential, models
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def show_images_by(file_path, file_name):
    images = []
    for img_path in glob.glob(file_path):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20, 10), dpi=1500)
    columns = 5

    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

    plt.savefig(file_name)


#####################################################################################################################
# Load the data                                                                                                    ##
#####################################################################################################################

# show_images_by('dataset/shapes/circles/*.png', 'circles-triangles-classification/dataset/output/circles.png')
# show_images_by('dataset/shapes/squares/*.png', 'circles-triangles-classification/dataset/output/squares.png')
# show_images_by('dataset/shapes/triangles/*.png', 'circles-triangles-classification/dataset/output/triangles.png')

#####################################################################################################################

# train the CNN modal
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 3), activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# antes era 0.25
classifier.add(Dropout(rate=0.5))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# antes era 0.25
classifier.add(Dropout(rate=0.5))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(rate=0.5))  # antes era 0.25

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=512, activation='relu'))
# classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=4, activation='softmax'))

# show classifier detail
classifier.summary()

# Compiling the CNN
classifier.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Using ImageDataGenerator to read images from directories
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(28, 28),
                                                 batch_size=16,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(28, 28),
                                            batch_size=16,
                                            class_mode='categorical')

# Utilize callback to store the weights of the best model
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True)

# Now it's time to train the model, here we include the callback to our checkpointer
history = classifier.fit_generator(training_set,
                                   steps_per_epoch=100,
                                   epochs=20,
                                   callbacks=[checkpointer],
                                   validation_data=test_set,
                                   validation_steps=50)

# Load our classifier with the weights of the best model
# classifier.load_weights('best_weights.hdf5')
# classifier.save('shapes_cnn.h5')

# Displaying curves of loss and accuracy during training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Predicting new images
img_path = 'dataset/test_set/composed/drawing(6).png'

img = image.load_img(img_path, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

print(img_tensor.shape)

# predicting images
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = classifier.predict_classes(images, batch_size=10)
print("Predicted class is:", classes)

# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in classifier.layers[:12]]

# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=classifier.input,
                                outputs=layer_outputs)

# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()

layer_names = []
for layer in classifier.layers[:12]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
