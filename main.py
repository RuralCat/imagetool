
import warnings
from imagetool.model import Unet
from dataset import load_images_train_data
from dataset import patches_from_mask
from dataset import get_all_file
from dataset import fullfile
from dataset.augmentation import trainingset_augmentation
import os
import cv2

root_path = 'K:\BIGCAT\Projects\EM'

def rename():
    save_path = fullfile(root_path, 'data', 'masks')
    file_list = get_all_file(save_path)
    for file in file_list:
        os.rename(file, file.split('.')[0] + '.jpg')

def rgb2gray():
    im_dir = fullfile(root_path, 'data', 'images')
    save_dir = fullfile(root_path, 'data', 'gray images')
    imgs = get_all_file(im_dir)
    for imp in imgs:
        im = cv2.imread(imp)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(fullfile(save_dir, os.path.basename(imp).split('.')[0] + '.jpg'), im)


def augmentation():
    im_dir = fullfile(root_path, 'data', 'gray images')
    mask_dir = os.path.join(root_path, 'data', 'masks')
    output_dir = os.path.join(root_path, 'data/aug gray images')
    gt_output_dir = os.path.join(root_path, 'data/aug masks')
    trainingset_augmentation(im_dir,
                             256, 256,
                             samples=10000,
                             ground_truth_path=mask_dir,
                             output_dir=output_dir,
                             ground_truth_output_dir=gt_output_dir)

def aug0():
    data_path = 'K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain'
    if 1:
        imgs_path = fullfile(data_path, 'testing_groundtruth.tif')
        ims = cv2.imreadmulti(imgs_path)[1]
        for i in range(len(ims)):
            save_path = fullfile(data_path, 'test masks', 'image_{}.jpg'.format(i))
            cv2.imwrite(save_path, ims[i])
        print(len(ims))
    if 0:
        trainingset_augmentation(fullfile(data_path, 'gray images'),
                                 output_width=256,
                                 output_height=256,
                                 samples=10000,
                                 ground_truth_path=fullfile(data_path, 'masks'),
                                 output_dir=fullfile(data_path, 'aug gray images'),
                                 ground_truth_output_dir=fullfile(data_path, 'aug masks'))

def golgi_patches(config):
    # file path
    im_path = os.path.join(config.root_path, 'data/golgi.tif')
    mask_path = os.path.join(config.root_path, 'data/golgi_mask 0.tif')
    # create patches
    patches = patches_from_mask(im_path, mask_path, 10000, 51, label=0)
    #


from sklearn.model_selection import KFold, StratifiedKFold
import samples

if __name__ == '__main__':
    # rgb2gray()
    # rename()
    # augmentation()
    aug0()
    # data_path = 'K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain'

    """
    # ignore warnings
    warnings.filterwarnings("ignore")

    # configuration
    model_description = 'membrane_cs24_aug_inputbatch'
    config = ImageConfig(os.path.abspath('../'), model_description)
    config.lr = 3e-4
    config.operation = ConfigOpt.TRAIN
    config.images_dir = fullfile(data_path, 'aug gray images')
    config.masks_dir = fullfile(data_path, 'aug masks')

    config_dir = 'model_unet_membrane_cs24_aug_inputbatch_0129_1658'
    config = Config.load_config(os.path.abspath('../'), config_dir)

    # prepare data (augmentation)
    if config.operation == ConfigOpt.AUGMENTATION:
        augmentation(config)

    if isinstance(config.operation, ConfigOpt) and \
            config.operation is not ConfigOpt.AUGMENTATION:
        # create model
        model = Unet(config=config)

        # load data
        x, y = load_images_train_data(model, img_num=10000)

        # date generator
        date_gen = image_data_generator()

        # run model
        model.run_model(x, y, use_generator=False, date_gen=date_gen)
        config.save_config()
    # """





