
import numpy as np

# path
import os
ROOTPATH = os.path.abspath('../')
DATAPATH = os.path.join(ROOTPATH, 'dataset', 'normed images')
ANOTPATH = os.path.join(ROOTPATH, 'dataset', 'Annotations')

class Image:
    pass


class RawImage(Image):
    def __init__(self, data, mask_path=None):
        from imagetool.dataset import read_image
        # read image dataset
        if isinstance(data, str):
            self.image = read_image(data)
        else:
            self.image = data
        # read image mask
        self.mask = read_image(mask_path) if mask_path else None

    def plot(self):
        # plot_image(self.image)
        pass

    def augmentation(self):
        imgs = self._augmentation([self.image])
        masks = self._augmentation([self.mask])

        return imgs, masks

    def _augmentation(self, data_stack):
        from imagetool.dataset import augmentation as aug
        # rotate 45, 90, 135
        data_rot_stack = []
        for i in [45, 135]:
            d0= aug.augment_mask(data_stack, aug.rotate, angle=i)
            data_rot_stack.extend(d0)
        data_stack.extend(data_rot_stack)

        # flip: vertical flip, horizontal flip
        data_flip_stack = []
        for i in range(2):
            d0 = aug.augment_mask(data_stack, aug.flip, axis=i)
            data_flip_stack.extend(d0)
        # data_stack.extend(data_flip_stack)

        # crop at different scale & resize to destination size (like 572 X 572)
        data_crop_stack = []
        scales = [0.2, 0.3, 0.4]
        nums = [30, 30, 20]
        for scale, num in zip(scales, nums):
            sz = np.int32(np.floor(np.array(self.image.shape) * scale))
            d0 = aug.augment_mask(data_stack, aug.dense_crop,
                                  height=sz[0], width=sz[1], num=num)
            d0 = aug.augment_mask(d0, aug.resize, dst_size=(572, 572))
            data_crop_stack.extend(d0)
        data_stack = data_crop_stack

        return data_stack



    @property
    def is_single_channel(self):
        return len(self.image.shape) == 1


if __name__ == '__main__':
    # test
    pass

