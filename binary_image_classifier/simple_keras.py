from IPython import embed
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

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
        rescale=1./255,
    )
    return augmenter


def run_sample_image_augmentation(augmenter):
    img = load_img(TRAIN_PATH + 'cat/cat.0.jpg')
    x = img_to_array(img)                           # shape = (3, 374, 500)
    x = x.reshape((1,) + x.shape)
    i = 0
    for _ in augmenter.flow(x, batch_size=1, save_to_dir='preview_augmentation',
                            save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break


def get_train_data_generator(augmenter):
    train_generator = augmenter.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_X, IMG_Y),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    return train_generator


def get_model():

    model = Sequential()
    # Conv 1
    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMG_X, IMG_Y, NUM_CHANNELS))
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 2
    model.add(
        Conv2D(filters=32, kernel_size=(3, 3))
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv 3
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3))
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

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
    # run_sample_image_augmentation(augmenter)
    model = get_model()
    train_data_gen = get_train_data_generator(augmenter)
    train_model(model, train_data_gen)
