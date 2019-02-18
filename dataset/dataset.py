
import os
import numpy as np
import skimage.io as skio
import pickle
from tqdm import tqdm
import cv2

# lambda expression
fullfile = lambda path, *paths : os.path.join(path, *paths)
pathexists = lambda path: os.path.exists(path)

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_all_file(path):
    return [fullfile(path, f) for f in next(os.walk(path))[2]]

def get_all_folder(path):
    return [fullfile(path, f) for f in next(os.walk(path))[1]]

def find(exp, n=0):
    indices = np.transpose(np.nonzero(exp))
    if n > 0:
        indices = indices[:n]
    return indices

def read_image(im_path):
    img = skio.imread(im_path, plugin='pil')
    return img[0] if img.shape == (4, ) else img

def patches_from_mask(im, mask, num, patch_size, label=1):
    """
    patch is a square shape
    """
    # if input is string, try to read it
    if isinstance(im, str):
        im = read_image(im)
    if isinstance(mask, str):
        mask = read_image(mask)
    # image padding
    r = np.int((patch_size - 1) / 2)
    im = np.pad(im, ((r, r), (r, r), (0, 0)), 'symmetric')
    # find valid pixel indices
    indices = find(mask == label)
    # uniform space sampling
    step = np.int(indices.shape[0] / num)
    if step < 1: step = 1
    indices = indices[::step]
    patchs = np.zeros((indices.shape[0],) + (patch_size, patch_size), dtype=np.float32)
    for i in range(indices.shape[0]):
        ix = indices[i, 0]
        iy = indices[i, 1]
        patchs[i] = im[ix : ix + patch_size, iy: iy + patch_size]

    return patchs

def patches_from_image(im_path, patch_size, step=2):
    """
    we need use it to create patches for a test image
    if we use all pixels, a 1K * 1K image will generate 10GB size patches (patch size 51 X 51)
    so we use param:step to do downsampling
    """
    import itertools
    # read image
    im = read_image(im_path)
    # image padding
    r = np.int((patch_size - 1) / 2)
    im = np.pad(im, ((r, r), (r, r), (0, 0)), 'symmetric')
    # create sampling indices by param:step
    [s0, s1] = im.shape
    patches_num = np.int32(s0 / step + 1) * np.int32(s1 / step + 1)
    patches = np.zeros((patches_num, patch_size, patch_size), dtype=np.float32)
    cb = itertools.product(range(0, s0, step), range(0, s1, step))
    for i, ind in zip(range(patches_num), cb):
        patches[i] = im[ind[0] : ind[0] + patch_size, ind[1]: ind[1] + patch_size]

    return patches

def labelmap_from_patches(im_shape, labels, step=2):
    """
    :param labels : NXC, C - number of classes
    """
    # shape matching
    s0 = np.int32(im_shape[0] / step + 1)
    s1 = np.int32(im_shape[1] / step + 1)
    if len(labels) != s0 * s1:
        raise ValueError('image shape is not matched with labels')
    # process labels
    labels = np.argmax(labels, axis=1)
    #
    labels = np.reshape(labels, (s0, s1))
    label_map = cv2.resize(labels, im_shape)

    return  label_map


def image_data_generator():
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=0.2,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.05,
                                 zoom_range=0.05,
                                 horizontal_flip=True,
                                 fill_mode='reflect',
                                 validation_split=0.1)
    return datagen

def load_images(imgs_dir, output_shape, img_num=None):
    from imagetool.model.modelbase import data_format
    output_shape = output_shape[1:]
    if data_format == 'channels_first':
        dst_shape = output_shape[1:]
        channels = output_shape[0]
        axes = (2, 0, 1)
    else:
        dst_shape = output_shape[:-1]
        channels = output_shape[-1]
        axes = (0, 1, 2)
    # get images path
    imgs_path = get_all_file(imgs_dir)
    img_num = len(imgs_path) if img_num is None else min(img_num, len(imgs_path))
    imgs_path = imgs_path[:img_num]
    # read image stack
    imgs = np.zeros((img_num,) + output_shape, dtype=np.float32)
    with tqdm(total=img_num, desc='Processing', unit='Images') as p_bar:
        for path, i in zip(imgs_path, range(img_num)):
            # read image
            im = cv2.imread(path)
            # im = im / 255
            # resize
            if channels == 1:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, dst_shape)
            if len(im.shape) == 2:
                im = np.expand_dims(im, axis=-1)
            # change order
            im = im / 255
            imgs[i] = np.transpose(im, axes)
            # update progress bar
            p_bar.set_description('Processing {}'.format(os.path.basename(path)))
            p_bar.update(1)
    return imgs

def _image_normalization(x, mean_map=None):
    if mean_map is not None:
        if os.path.exists(mean_map):
            with open(mean_map, 'rb') as f:
                x_mean = pickle.load(f)
        else:
            x_mean = np.mean(x, axis=0)
            with open(mean_map, 'wb') as f:
                pickle.dump(x_mean, f)
    x -= x_mean
    return x

def img_norm(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x)
        std = np.std(x)
    return (x - mean) / std

def load_images_train_data(model, img_num=None):
    # get image & mask shape
    from imagetool.model.modelbase import ModelBase
    from config import ImageConfig
    assert isinstance(model, ModelBase)
    input_shape = model.model.input_shape
    output_shape = model.model.output_shape

    # load
    config = model.config
    assert isinstance(config, ImageConfig)
    train_x = load_images(config.images_dir, input_shape, img_num)
    train_y = load_images(config.masks_dir, output_shape, img_num)

    # normlize
    # train_x = _image_normalization(train_x, config.mean_map)

    # shuffle
    np.random.seed(20190121)
    idx = np.random.permutation(train_x.shape[0])
    train_x = train_x[idx]
    train_y = train_y[idx]

    return train_x, train_y

def load_image_test_data(model, imgs_dir, nb_imgs=None):
    # get image shape
    from imagetool.model.modelbase import ModelBase
    from config import ImageConfig
    assert isinstance(model, ModelBase)
    input_shape = model.model.input_shape
    # load
    test_x = load_images(imgs_dir, input_shape, nb_imgs)
    # nomlization
    assert isinstance(model.config, ImageConfig)
    # test_x = _image_normalization(test_x, model.config.mean_map)

    return test_x


if __name__ == '__main__':
   import tkinter as tk
   root = tk.Tk()
   root.after()
