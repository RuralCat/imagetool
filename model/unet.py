
from imagetool.model.modelbase import *
from imagetool.config import ImageConfig


def _unet(channel_size=24):
   # image input
   if data_format == 'channels_first':
       img_input = Input(shape=(1, 256, 256))
       channel_axis = 1
   else:
       img_input = Input(shape=(256, 256, 1))
       channel_axis = -1
   x =BatchNormalization(axis=channel_axis,
                         momentum=0.995,
                         scale=False)(img_input)

   # channel size
   cs = channel_size

   # down sampling
   c1, p1 = ConvBlock0(x, cs, name='conv_block0')
   c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
   c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
   c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
   c5= ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

   # upsampleing
   upconv0 = UpConvBlock0(c4, c5, 8 * cs, name='upconv_block0')
   upconv1 = UpConvBlock0(c3, upconv0, 4 * cs, name='upconv_block1')
   upconv2 = UpConvBlock0(c2, upconv1, 2 * cs, name='upconv_block2')
   upconv3 = UpConvBlock0(c1, upconv2, cs, name='upconv_block3')

   # mask output
   mask_output = Conv2D(1, (1,1), data_format=data_format,
                        activation='sigmoid', name='output')(upconv3)

   return Model(img_input, mask_output)

class Unet(ModelBase):
    def __init__(self, config=None):
        assert isinstance(config, ImageConfig)
        model = _unet(config.channel_size)
        ModelBase.__init__(self, model, config)

class InceptionUnet(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(572, 572, 1,))

        # channel size
        cs = 32

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # upsampleing
        upconv0 = UpConvBlock0(c4, c5, 4, 8 * cs, name='upconv_block0')
        upconv1 = UpConvBlock0(c3, upconv0, 16, 4 * cs, name='upconv_block1')
        upconv2 = UpConvBlock0(c2, upconv1, 40, 2 * cs, name='upconv_block2')
        # upconv3 = UpConvBlock0(c1, upconv2, 88, cs, name='upconv_block3')
        upconv3 = UpConvBlock1(upconv2, cs, name='upconv_block3')

        # mask output
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(upconv3)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)

class Unetv2(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(572, 572, 3,))

        # channel size
        cs = 24

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c1_e = ConvBlock0(c1, cs, False, 'conv_block0_e')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c2_e = ConvBlock0(c2, 2 * cs, False, 'conv_block1_e')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c3_e = ConvBlock0(c3, 4 * cs, False, 'conv_block2_e')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c4_e = ConvBlock0(c4, 8 * cs, False, 'conv_block3_e')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # upsampleing
        upconv0 = UpConvBlock0(c4_e, c5, 2, 8 * cs, name='upconv_block0')
        upconv1 = UpConvBlock0(c3_e, upconv0, 14, 4 * cs, name='upconv_block1')
        upconv2 = UpConvBlock0(c2_e, upconv1, 38, 2 * cs, name='upconv_block2')
        upconv3 = UpConvBlock0(c1_e, upconv2, 86, cs, name='upconv_block3')

        # mask output
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(upconv3)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)

class TriangularNet(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(572, 572, 3, ))
        cs = 24

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # first upsampling
        fir_upconv0 = UpConvBlock0(c1, c2, 4, cs, name='first_upconv_block0')
        fir_upconv1 = UpConvBlock0(c2, c3, 4, 2 * cs, name='first_upconv_block1')
        fir_upconv2 = UpConvBlock0(c3, c4, 4, 4 * cs, name='first_upconv_block2')
        fir_upconv3 = UpConvBlock0(c4, c5, 4, 8 * cs, name='first_upconv_block3')

        # second upsampling
        sec_upconv0 = UpConvBlock0(fir_upconv0, fir_upconv1, 10, cs, name='second_upconv_block0')
        sec_upconv1 = UpConvBlock0(fir_upconv1, fir_upconv2, 10, 2 * cs, name='second_upconv_block1')
        sec_upconv2 = UpConvBlock0(fir_upconv2, fir_upconv3, 10, 4 * cs, name='second_upconv_block2')

        # third upsampling
        tir_upconv0 = UpConvBlock0(sec_upconv0, sec_upconv1, 22, cs, name='third_upconv_block0')
        tir_upconv1 = UpConvBlock0(sec_upconv1, sec_upconv2, 22, 2 * cs, name='third_upconv_block1')

        # final & output
        fou_upconv0 = UpConvBlock0(tir_upconv0, tir_upconv1, 46, cs, name='fou_upconv_block0')
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(fou_upconv0)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)

class Unet_padding(ModelBase):
    def __init__(self):
        # image input
        img_input = Input(shape=(512, 512, 1,))

        # channel size
        cs = 24

        # down sampling
        c1, p1 = ConvBlock0(img_input, cs, name='conv_block0')
        c2, p2 = ConvBlock0(p1, 2 * cs, name='conv_block1')
        c3, p3 = ConvBlock0(p2, 4 * cs, name='conv_block2')
        c4, p4 = ConvBlock0(p3, 8 * cs, name='conv_block3')
        c5 = ConvBlock0(p4, 16 * cs, pooling=False, name='conv_block4')

        # upsampleing
        upconv0 = UpConvBlock0(c4, c5, 0, 8 * cs, name='upconv_block0')
        upconv1 = UpConvBlock0(c3, upconv0, 0, 4 * cs, name='upconv_block1')
        upconv2 = UpConvBlock0(c2, upconv1, 0, 2 * cs, name='upconv_block2')
        upconv3 = UpConvBlock0(c1, upconv2, 0, cs, name='upconv_block3')

        # mask output
        mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='output')(upconv3)
        # mask_output = Lambda(Squeeze)(mask_output)

        self.model = Model(img_input, mask_output)

class Unetv3(ModelBase):
    def __init__(self):
        pass

if __name__ == '__main__':
    unet = _unet(24)
    unet.summary()
