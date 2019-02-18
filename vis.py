
from config import ImageConfig
from config import ConfigOpt
import os
from imagetool.model import Unet
from dataset import load_image_test_data
from dataset import load_images_train_data
from matplotlib import pyplot as plt
import cv2
from dataset import get_all_file
from dataset import fullfile

test_images_dir = 'K:\BIGCAT\Projects\EM\\data\Other data\EM CA1 hippocampus region of brain\\test images'
test_gt_dir = 'K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain\\test masks'
train_images_dir = 'K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain\gray images'
gt_dir = 'K:\BIGCAT\Projects\EM\data\Other data\EM CA1 hippocampus region of brain\masks'

root_path = os.path.abspath('../')
config_dir = 'model_unet_membrane_cs24_aug_inputbatch_0129_1658'
config = ImageConfig.load_config(root_path, config_dir=config_dir)
assert isinstance(config, ImageConfig)
config.images_dir = train_images_dir
config.masks_dir = gt_dir
config.operation = ConfigOpt.PREDICT

# create model
model = Unet(config)


test_x = load_image_test_data(model, test_images_dir)
print(test_x.shape)
train_x, train_y = load_images_train_data(model)

pred_test_y = model.run_model(test_x)
pre_train_y = model.run_model(train_x)
mean_iou = model.model.evaluate(train_x, train_y)

fig, axes = plt.subplots(5, 3, figsize=(15, 15))

inds = [21, 45, 67]
for i in range(3):
    ind = inds[i]
    axes[i,0].imshow(train_x[ind][0], cmap="gray")
    axes[i,0].axis("off")
    axes[i,0].set_title("train volume - {}".format(ind))
    axes[i,1].imshow(train_y[ind][0] > 0.5, cmap="gray")
    axes[i,1].axis("off")
    axes[i,1].set_title("threshold 0.5")
    axes[i,2].imshow(train_y[ind][0], cmap="gray")
    axes[i,2].axis("off")
    axes[i,2].set_title("ground truth")

inds = [17, 79]
for i in [3, 4]:
    ind = inds[i-3]
    axes[i,0].imshow(test_x[ind][0], cmap="gray")
    axes[i,0].axis("off")
    axes[i,0].set_title("test volume - {}".format(ind))
    axes[i,1].imshow(pred_test_y[ind][0] > 0.5, cmap="gray")
    axes[i,1].axis("off")
    axes[i,1].set_title("threshold 0.5")
    axes[i,2].imshow(pred_test_y[ind][0] > 0.7, cmap="gray")
    axes[i,2].axis("off")
    axes[i,2].set_title("threshold 0.7")

fig.tight_layout()
plt.show()