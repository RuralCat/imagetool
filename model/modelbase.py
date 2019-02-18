from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from imagetool.config import Config
from imagetool.config import ConfigOpt

data_format = 'channels_first'
# data_format = K.image_data_format()
channel_axis = 1 if data_format == 'channels_first' else -1
import tkinter

class ModelBase:
    def __init__(self, model, config=None):
        # model
        assert isinstance(model, Model)
        self.model = model
        # config
        if config is not None:
            assert isinstance(config, Config)
            self.config = config
        else:
            self.config = Config()
        # set placeholder
        self.x = None
        self.y = None

    def _compile(self):
        self.model.compile(optimizer=self.config.optimizer,
                           loss=self.config.loss,
                           metrics=self.config.metrics)

    def _reset_param(self, param, value):
        if value is not None:
            self.config.set_param(param, value)

    def _fit(self, callbacks):
        history = self.model.fit(self.x, self.y,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       verbose=1,
                       shuffle=True,
                       validation_split=self.config.validation_split,
                       callbacks=callbacks)
        return history

    def _fit_generator(self, data_gen, callbacks):
        assert isinstance(data_gen, ImageDataGenerator)
        data_gen_flow = data_gen.flow(self.x, self.y, batch_size=self.config.batch_size)
        if hasattr(self.config, 'steps_per_epoch'):
            steps_per_epoch = self.config.steps_per_epoch
        else:
            steps_per_epoch = self.num_samples / self.config.batch_size
        history = self.model.fit_generator(data_gen_flow,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=self.config.epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=data_gen_flow,
                                 validation_steps=20)
        return history

    def _dump_data(self, x, y=None, **kwargs):
        self.x = x
        self.y = y
        self.num_samples = x.shape[0]

    def _train(self, use_generator=False, **kwargs):
        # create callbacks
        callbacks = self.config.learning_callbacks
        # train
        assert isinstance(self.model, Model)
        if use_generator:
            data_gen = kwargs.get('data_gen', None)
            history = self._fit_generator(data_gen, callbacks)
        else:
            history = self._fit(callbacks)
        # save weights
        self.model.save_weights(self.config.weights_path)
        # save model
        self.model.save(self.config.model_path)

    def _predict(self):
        # load weights
        self.model.load_weights(self.config.weights_path)
        # predict
        y = self.model.predict(self.x)
        return y

    def run_model(self, x, y=None, lr=None, weights=None, use_generator=False, **kwargs):
        # dump data
        self._dump_data(x, y, **kwargs)

        # reset paras
        self._reset_param('lr', lr)
        self._reset_param('weights_path', weights)

        # compile model
        self._compile()

        # run model
        op = self.config.operation
        assert isinstance(op, ConfigOpt)
        if op == ConfigOpt.TRAIN:
            self._train(use_generator, **kwargs)
        elif op == ConfigOpt.PREDICT:
             return self._predict()
        elif op == ConfigOpt.EVALUATE:
            pass


def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False, name=''):
    """
    Utility function to apply conv + BN.
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    padding = 'same'
    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Conv2D(nb_filter,
               (num_row, num_col),
               data_format=data_format,
               name=name,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_regularizer=regularizers.l2(0.00004),
               kernel_initializer=initializers.VarianceScaling(scale=2.0,
                                                               mode='fan_in',
                                                               distribution='normal',
                                                               seed=None))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def ConvBlock0(input_layer, nb_filter, pooling=True, name=''):
    c0 = conv2d_bn(input_layer, nb_filter, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, nb_filter, 3, 3, padding='valid', name='{}_1'.format(name))
    if pooling:
        p0 = MaxPooling2D(pool_size=(2,2), data_format=data_format)(c1)
        return c1, p0
    else:
        return c1

def Squeeze(x):
    return K.squeeze(x, axis=-1)

def UpConvBlock0(x0, x1, nb_filter, name=None, bCat=True):
    # deconvolution
    layer = Conv2DTranspose(nb_filter,
                            kernel_size=(2, 2),
                            strides=(2,2),
                            data_format=data_format)
    output_shape = layer.compute_output_shape(x1.shape)
    x1 = layer(x1)
    if bCat:
        # compute cropping nb_filter
        i = 2 if data_format == 'channels_first' else 1
        c0 = np.int((x0.shape[i].value - output_shape[i].value) / 2)
        c1 = np.int((x0.shape[i+1].value - output_shape[i+1].value) / 2)
        # crop to match nb_filter
        x0 = Cropping2D((c0, c1), data_format=data_format)(x0)
        # concatenation
        x = Concatenate(axis=channel_axis)([x0, x1])
    else:
        x = x1
    # two convolution
    x = conv2d_bn(x, nb_filter, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(x, nb_filter, 3, 3, padding='valid', name='{}_1'.format(name))

    return c1

def UpConvBlock1(layer, size, name=None):
    upconv0 = Conv2DTranspose(size, (2, 2), strides=(2, 2))(layer)
    c0 = conv2d_bn(upconv0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    return c1

def UpConvBlock2(layer, size, name=''):
    upconv0 = Conv2DTranspose(size, (2, 2), strides=(2, 2))(layer)
    c0 = conv2d_bn(upconv0, size, 3, 3, padding='valid', name='{}_0'.format(name))
    c1 = conv2d_bn(c0, size, 3, 3, padding='valid', name='{}_1'.format(name))
    c2 = MaxPooling2D(pool_size=(2,2))(c1)

    return c2


if __name__ == '__main__':
    # x,y = Unet.load_training_data()
    pass

