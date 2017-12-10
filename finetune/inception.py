from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

NUM_CLASSES = 2


def batch_generator():
    pass

# base pre-trained model
base_model = InceptionV3(include_top=False, weights='imagenet')

# Global
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Fully connected layer
x = Dense(units=1024, activation='relu')(x)
# Logistic softmax layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Train only the top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Train the model
model.fit_generator()
