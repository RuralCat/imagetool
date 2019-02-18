
from imagetool.model.modelbase import *

def TernaryModel0(bdropout = True):
    # input
    input = Input(shape=(51, 51, 3, ))
    # conv 1
    conv1 = Conv2D(25, (4,4), activation='relu')(input)
    if bdropout: conv1 = Dropout(0.1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    # conv 2
    conv2 = Conv2D(50, (5,5), activation='relu')(conv1)
    if bdropout: conv2 = Dropout(0.2)(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    # conv 3
    conv3 = Conv2D(80, (6,6), activation='relu')(conv2)
    if bdropout: conv3 = Dropout(0.25)(conv3)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv3)
    # dense
    flat = Flatten()(conv3)
    dense1 = Dense(1024, activation='relu')(flat)
    if bdropout: dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(1024, activation='relu')(dense1)
    if bdropout: dense2 = Dropout(0.5)(dense2)
    # output
    output = Dense(3, activation='softmax')(dense2)
    # create model
    model = Model(inputs=input, outputs=output)

    return model

def convblock(x, size, channels, name=''):
    conv = conv2d_bn(x, channels, size, size, name=name)
    conv = MaxPooling2D()(conv)
    return conv

def denseblock(x, nb_filter, name=''):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Dense(nb_filter,
              name=name,
              use_bias=False,
              kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.99, scale=False)(x)
    x = Activation('relu')(x)
    return x

def _tenarymodel():
    # input
    input = Input(shape=(51, 51, 3,))
    # conv1
    conv1 = convblock(input, 4, 25, 'conv_1')
    # conv2
    conv2 = convblock(conv1, 5, 50, 'conv_2')
    # conv1
    conv3 = convblock(conv2, 6, 80, 'conv_3')
    # dense
    flat = Flatten()(conv3)
    dense1 = denseblock(flat, 1024, name='dense_1')
    dense2 = denseblock(dense1, 1024, name='dense_1')
    dense2 = Dropout(0.5)(dense2)
    # output
    output = Dense(3, activation='softmax')(dense2)
    # create model
    model = Model(inputs=input, outputs=output)

    return model

class TenaryModel(ModelBase):
    def __init__(self, config=None):
        from config import ImageConfig
        assert isinstance(config, ImageConfig)
        model = _tenarymodel()
        ModelBase.__init__(model, config)

