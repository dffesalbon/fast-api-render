from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Dropout

import tensorflow as tf

def base_model():
    # Load the pretained model
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max',
    )

    pretrained_model.trainable = False

    augment = get_augmentation()

    # Existing model structure
    inputs = pretrained_model.input
    x = augment(inputs)
    x = Dense(128, activation='relu')(pretrained_model.output)
    x = Dropout(0.45)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.45)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model



def get_augmentation():
    # Data Augmentation Step
    augment = tf.keras.Sequential([
        layers.Resizing(224,224),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        ])
    return augment


# def process_image():
#     test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

#     test_images = test_generator.flow_from_dataframe(
#         dataframe=test_df,
#         x_col='img_path',
#         y_col='label',
#         target_size=TARGET_SIZE,
#         color_mode='rgb',
#         class_mode='categorical',
#         batch_size=BATCH_SIZE,
#         shuffle=False
#         )