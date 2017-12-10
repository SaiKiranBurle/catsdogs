from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

TRAIN_PATH = '/Users/sai/dev/datasets/catsdogs-kaggle/data2/train/'

# Constants
NUM_CHANNELS = 3
IMG_X = 150
IMG_Y = 150

BATCH_SIZE = 16
TOTAL_NUM_IMAGES = 25000


def get_train_data_augmenter():
    # real time image augmentation
    augmenter = ImageDataGenerator(
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        # rescale=1./255,
        preprocessing_function=preprocess_input
    )
    return augmenter


def get_train_data_generator(augmenter):
    train_generator = augmenter.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_X, IMG_Y),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    return train_generator


def get_model():
    # base pre-trained model
    base_model = InceptionV3(include_top=False, weights='imagenet')

    # Global
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(units=1024, activation='relu')(x)
    # Logistic softmax layer
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, data_gen):
    model.fit_generator(
        data_gen,
        steps_per_epoch=TOTAL_NUM_IMAGES // BATCH_SIZE,
        epochs=50
    )

if __name__ == "__main__":
    augmenter = get_train_data_augmenter()
    model = get_model()
    train_data_gen = get_train_data_generator(augmenter)
    train_model(model, train_data_gen)
