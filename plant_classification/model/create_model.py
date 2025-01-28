import keras
from keras.models import Sequential
from keras import layers
from keras import activations

def create_simple_model(image_shape, num_classes):
    """
    A function to create a simple CNN model composed of 3 convolution layers
    and a fully-connected layer with softmax classifier.

    Args:
        tuple image_shape: Shape of the images to be trained on
        int num_classes: Number of classes to classify

    Returns:
        keras model
    """
    keras.backend.clear_session()

    model = Sequential()
    model.add(keras.Input(shape=image_shape))

    # add convolution at increasing filter size
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # add fully-connected layer
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # softmax classifier
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def compile_model(model):
    """
    A function to compile a model

    Args:
        keras.model model: Model to compile

    Returns:
        None
    """
    opt = keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
