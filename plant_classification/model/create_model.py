import keras
from keras.models import Sequential
from keras import layers
from keras import activations

def create_model(model_type, image_shape, num_classes, print_summary=0):
    """
    A function to create a CNN model, either simple or pretrained.

    Args:
        str model_type: The type of model to run, either 'simple' or 'pretrained'
        tuple image_shape: Shape of the images to be trained on
        int num_classes: Number of classes to classify
        bool print_summary: Boolean indicating to print model summary (default: 0)

    Returns:
        tf.keras.Model
    """
    if model_type not in ['simple','pretrained']:
        raise ValueError("model_type must be 'simple' or 'pretrained'.")

    if model_type == 'simple':
        model = create_simple_model(image_shape, num_classes)
    else:
        model = init_pretrained_model(image_shape, num_classes)

    if print_summary:
        model.summary()

    return model


def create_simple_model(image_shape, num_classes):
    """
    A function to create a simple CNN model composed of 3 convolution layers
    and a fully-connected layer with softmax classifier.

    Args:
        tuple image_shape: Shape of the images to be trained on
        int num_classes: Number of classes to classify

    Returns:
        tf.keras.Model
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


def init_pretrained_model(image_shape, num_classes):
    """
    A function to initialised the pre-trained MobileNetV2 model and specialise
    it to classifying our data.

    Args:
        tuple image_shape: Shape of the images to be trained on
        int num_classes: Number of classes to classify

    Returns:
        tf.keras.Model
    """

    from tensorflow.keras.applications import MobileNetV2

    # call for MobileNetV2, without the top classification layers
    base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model.trainable = False # don't alter base layers

    # Add specialised training layers for our problem
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model



def compile_model(model, opt, loss_func, metrics):
    """
    A function to compile a model

    Args:
        tf.keras.Model model: Model to compile
        tf.keras.optimizers opt: Optimizer type
        tf.keras.losses loss_func: Loss function
        list of str metrics: The metrics to minimise over

    Returns:
        Compiled tf.keras.Model
    """

    model.compile(optimizer=opt,
                 loss=loss_func,
                 metrics=['accuracy'])
