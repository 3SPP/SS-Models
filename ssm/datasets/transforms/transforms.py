import numpy as np
import random
import cv2
import math
from PIL import Image
import ssm.datasets.transforms.function as f


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """
    def __init__(self, transforms, bands=[1, 2, 3]):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.bands = bands

    def __call__(self, im1, im2=None, label=None):
        """
        Args:
            im1 (np.ndarray): It is either image path or image object.
            im2 (np.ndarray): It is either image path or image object.
            label (np.ndarray): It is either label path or label ndarray.
        Returns:
            (tuple). A tuple including image1, image2, and label after transformation.
        """
        im1 = im1.astype('float32')
        if im2 is not None:
            im2 = im2.astype('float32')
        if label is not None:
            label = label.astype('float32')
        for op in self.transforms:
            outputs = op(im1, im2, label)
            im1, im2, label = outputs
        im1 = np.transpose(im1, (2, 0, 1))
        if im2 is not None:
            im2 = np.transpose(im2, (2, 0, 1))
        return (im1, im2, label)


class RandomHorizontalFlip:
    """
    Flip an image horizontally with a certain probability.
    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2=None, label=None):
        if random.random() < self.prob:
            im1 = f.horizontal_flip(im1)
            if im2 is not None:
                im2 = f.horizontal_flip(im2)
            if label is not None:
                label = f.horizontal_flip(label)
        return (im1, im2, label)


class RandomVerticalFlip:
    """
    Flip an image vertically with a certain probability.
    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            im1 = f.vertical_flip(im1)
            if im2 is not None:
                im2 = f.horizontal_flip(im2)
            if label is not None:
                label = f.vertical_flip(label)
        return (im1, im2, label)


class Resize:
    """
    Resize an image.
    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".
    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im1, im2, label=None):
        if len(im1.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im1= f.resize(im1, self.target_size, self.interp_dict[interp])
        if im2 is not None:
            im2= f.resize(im2, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = f.resize(label, self.target_size, cv2.INTER_NEAREST)
        return (im1, im2, label)


class ResizeByLong:
    """
    Resize the long side of an image to given size, and then scale the other side proportionally.
    Args:
        long_size (int): The target size of long side.
    """
    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im1, im2, label=None):
        im1 = f.resize_long(im1, self.long_size)
        if im2 is not None:
            im2 = f.resize_long(im2, self.long_size)
        if label is not None:
            label = f.resize_long(label, self.long_size, cv2.INTER_NEAREST)
        return (im1, im2, label)


class ResizeByShort:
    """
    Resize the short side of an image to given size, and then scale the other side proportionally.
    Args:
        short_size (int): The target size of short side.
    """
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, im1, im2, label=None):
        im1 = f.resize_short(im1, self.short_size)
        if im2 is not None:
            im2 = f.resize_short(im2, self.short_size)
        if label is not None:
            label = f.resize_short(label, self.short_size, cv2.INTER_NEAREST)
        return (im1, im2, label)


class LimitLong:
    """
    Limit the long edge of image.
    If the long edge is larger than max_long, resize the long edge
    to max_long, while scale the short edge proportionally.
    If the long edge is smaller than min_long, resize the long edge
    to min_long, while scale the short edge proportionally.
    Args:
        max_long (int, optional): If the long edge of image is larger than max_long,
            it will be resize to max_long. Default: None.
        min_long (int, optional): If the long edge of image is smaller than min_long,
            it will be resize to min_long. Default: None.
    """
    def __init__(self, max_long=None, min_long=None):
        if max_long is not None:
            if not isinstance(max_long, int):
                raise TypeError(
                    "Type of `max_long` is invalid. It should be int, but it is {}"
                    .format(type(max_long)))
        if min_long is not None:
            if not isinstance(min_long, int):
                raise TypeError(
                    "Type of `min_long` is invalid. It should be int, but it is {}"
                    .format(type(min_long)))
        if (max_long is not None) and (min_long is not None):
            if min_long > max_long:
                raise ValueError(
                    '`max_long should not smaller than min_long, but they are {} and {}'
                    .format(max_long, min_long))
        self.max_long = max_long
        self.min_long = min_long

    def __call__(self, im1, im2, label=None):
        h, w = im1.shape[0], im1.shape[1]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long
        if target != long_edge:
            im1 = f.resize_long(im1, target)
            if im2 is not None:
                im2 = f.resize_long(im2, target)
            if label is not None:
                label = f.resize_long(label, target, cv2.INTER_NEAREST)
        return (im1, im2, label)


class ResizeRangeScaling:
    """
    Resize the long side of an image into a range, and then scale the other side proportionally.
    Args:
        min_value (int, optional): The minimum value of long side after resize. Default: 400.
        max_value (int, optional): The maximum value of long side after resize. Default: 600.
    """
    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(
                                 min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im1, im2, label=None):
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        im1 = f.resize_long(im1, random_size, cv2.INTER_LINEAR)
        if im2 is not None:
            im2 = f.resize_long(im2, random_size, cv2.INTER_LINEAR)
        if label is not None:
            label = f.resize_long(label, random_size, cv2.INTER_NEAREST)
        return (im1, im2, label)


class ResizeStepScaling:
    """
    Scale an image proportionally within a range.
    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.
    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """
    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im1, im2, label=None):
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor
        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)
        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * im1.shape[1]))
        h = int(round(scale_factor * im1.shape[0]))
        im1 = f.resize(im1, (w, h), cv2.INTER_LINEAR)
        if im2 is not None:
            im2 = f.resize(im2, (w, h), cv2.INTER_LINEAR)
        if label is not None:
            label = f.resize(label, (w, h), cv2.INTER_NEAREST)
        return (im1, im2, label)


class Normalize:
    """
    Normalize an image.
    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].
    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im1, im2, label=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im1 = f.normalize(im1, mean, std)
        if im2 is not None:
            im2 = f.normalize(im2, mean, std)
        return (im1, im2, label)


class Padding:
    """
    Add bottom-right padding to a raw image or annotation image.
    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """
    def __init__(self,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of target_size is invalid. It should be list or tuple, now is {}"
                .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        im_height, im_width = im1.shape[0], im1.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'The size of image should be less than `target_size`, but the size of image ({}, {}) is larger than `target_size` ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im1 = cv2.copyMakeBorder(im1, 0, pad_height, 0, pad_width,
                                     cv2.BORDER_CONSTANT, value=self.im_padding_value)
            if im2 is not None:
                im2 = cv2.copyMakeBorder(im2, 0, pad_height, 0, pad_width,
                                         cv2.BORDER_CONSTANT, value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(label, 0, pad_height, 0, pad_width, 
                                           cv2.BORDER_CONSTANT, value=self.label_padding_value)
        return (im1, im2, label)


class PaddingByAspectRatio:
    """
    Args:
        aspect_ratio (int|float, optional): The aspect ratio = width / height. Default: 1.
    """

    def __init__(self,
                 aspect_ratio=1,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.aspect_ratio = aspect_ratio
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        img_height = im1.shape[0]
        img_width = im1.shape[1]
        ratio = img_width / img_height
        if ratio == self.aspect_ratio:
            return (im1, im2, label)
        elif ratio > self.aspect_ratio:
            img_height = int(img_width / self.aspect_ratio)
        else:
            img_width = int(img_height * self.aspect_ratio)
        padding = Padding((img_width, img_height),
                          im_padding_value=self.im_padding_value,
                          label_padding_value=self.label_padding_value)
        return padding(im1, im2, label)


class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.
    Args:
        crop_size (tuple, optional): The target cropping size. Default: (512, 512).
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """
    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'Type of `crop_size` is list or tuple. It should include 2 elements, but it is {}'
                    .format(crop_size))
        else:
            raise TypeError(
                "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]
        img_height = im1.shape[0]
        img_width = im1.shape[1]
        if img_height == crop_height and img_width == crop_width:
            return (im1, im2, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im1 = cv2.copyMakeBorder(im1, 0, pad_height, 0, pad_width,
                                         cv2.BORDER_CONSTANT, value=self.im_padding_value)
                if im2 is not None:
                    im2 = cv2.copyMakeBorder(im2, 0, pad_height, 0, pad_width,
                                             cv2.BORDER_CONSTANT, value=self.im_padding_value)
                if label is not None:
                    label = cv2.copyMakeBorder(label, 0, pad_height, 0, pad_width,
                                               cv2.BORDER_CONSTANT, value=self.label_padding_value)
                img_height = im1.shape[0]
                img_width = im1.shape[1]
            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)
                im1 = im1[h_off:(crop_height + h_off), w_off:(w_off + crop_width), :]
                if im2 is not None:
                    im2 = im2[h_off:(crop_height + h_off), w_off:(w_off + crop_width), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(w_off + crop_width)]
        return (im1, im2, label)


class ScalePadding:
    """
        Add center padding to a raw image or annotation image,then scale the
        image to target size.
        Args:
            target_size (list|tuple, optional): The target size of image. Default: (512, 512).
            im_padding_value (list, optional): The padding value of raw image.
                Default: [127.5, 127.5, 127.5].
            label_padding_value (int, optional): The padding value of annotation image. Default: 255.
        Raises:
            TypeError: When target_size is neither list nor tuple.
            ValueError: When the length of target_size is not 2.
    """
    def __init__(self,
                 target_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        height = im1.shape[0]
        width = im1.shape[1]
        new_im1 = np.zeros(
            (max(height, width), max(height, width), 3)) + self.im_padding_value
        if im2 is not None:
            new_im2 = np.zeros(
                (max(height, width), max(height, width), 3)) + self.im_padding_value
        if label is not None:
            new_label = np.zeros((max(height, width), max(
                height, width))) + self.label_padding_value
        if height > width:
            padding = int((height - width) / 2)
            new_im1[:, padding:padding + width, :] = im1
            if im2 is not None:
                new_im2[:, padding:padding + width, :] = im2
            if label is not None:
                new_label[:, padding:padding + width] = label
        else:
            padding = int((width - height) / 2)
            new_im1[padding:padding + height, :, :] = im1
            if im2 is not None:
                new_im2[padding:padding + height, :, :] = im2
            if label is not None:
                new_label[padding:padding + height, :] = label
        im1 = np.uint8(new_im1)
        im1 = f.resize(im1, self.target_size, interp=cv2.INTER_CUBIC)
        if im2 is not None:
            im2 = np.uint8(new_im2)
            im2 = f.resize(im2, self.target_size, interp=cv2.INTER_CUBIC)
        if label is not None:
            label = np.uint8(new_label)
            label = f.resize(
                label, self.target_size, interp=cv2.INTER_CUBIC)
        return (im1, im2, label)



class RandomNoise:
    """
    Superimposing noise on an image with a certain probability.
    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.5.
        max_sigma(float, optional): The maximum value of standard deviation of the distribution.
            Default: 10.0.
    """
    def __init__(self, prob=0.5, max_sigma=10.0):
        self.prob = prob
        self.max_sigma = max_sigma

    def __call__(self, im1, im2, label=None):
        if random.random() < self.prob:
            mu = 0
            sigma = random.random() * self.max_sigma
            im1 = np.array(im1, dtype=np.float32)
            im1 += np.random.normal(mu, sigma, im1.shape)
            im1[im1 > 255] = 255
            im1[im1 < 0] = 0
            if im2 is not None:
                im2 = np.array(im2, dtype=np.float32)
                im2 += np.random.normal(mu, sigma, im2.shape)
                im2[im2 > 255] = 255
                im2[im2 < 0] = 0
        return (im1, im2, label)


class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.
    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            random represents randomly selected from above.
            Default: gaussian.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

    def __call__(self, im1, im2, label=None):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im1 = f.random_blur(im1, radius, self.blur_type)
                if im2 is not None:
                    im2 = f.random_blur(im2, radius, self.blur_type)
        im1 = np.array(im1, dtype='float32')
        if im2 is not None:
            im2 = np.array(im2, dtype='float32')
        return (im1, im2, label)



class RandomRotation:
    """
    Rotate an image randomly with padding.
    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """
    def __init__(self,
                 max_rotation=15,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        if self.max_rotation > 0:
            (h, w) = im1.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])
            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))
            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            im1 = cv2.warpAffine(im1, r, dsize=dsize, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=self.im_padding_value)
            if im2 is not None:
                im2 = cv2.warpAffine(im2, r, dsize=dsize, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(label, r, dsize=dsize, flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=self.label_padding_value)
        return (im1, im2, label)


class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.
    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """
    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im1, im2, label=None):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im1.shape[0]
            img_width = im1.shape[1]
            for _ in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)
                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp
                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)
                    im1 = im1[h1:(h1 + dh), w1:(w1 + dw), :]
                    im1 = cv2.resize(
                        im1, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if im2 is not None:
                        im2 = im2[h1:(h1 + dh), w1:(w1 + dw), :]
                        im2 = cv2.resize(
                            im2, (img_width, img_height),
                            interpolation=cv2.INTER_LINEAR)
                    if label is not None:
                        label = label[h1:(h1 + dh), w1:(w1 + dw)]
                        label = cv2.resize(
                            label, (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        return (im1, im2, label)


class RandomDistort:
    """
    Distort an image with random configurations.
    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """
    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, im1, im2, label=None):
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            f.brightness, f.contrast, f.saturation,
            f.hue, f.sharpness
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        im1 = im1.astype('uint8')
        im1 = Image.fromarray(im1)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im1
            if np.random.uniform(0, 1) < prob:
                im1 = ops[id](**params)
        im1 = np.asarray(im1).astype('float32')
        if im2 is not None:
            im2 = im2.astype('uint8')
            im2 = Image.fromarray(im2)
            for id in range(len(ops)):
                params = params_dict[ops[id].__name__]
                prob = prob_dict[ops[id].__name__]
                params['im'] = im2
                if np.random.uniform(0, 1) < prob:
                    im2 = ops[id](**params)
            im2 = np.asarray(im2).astype('float32')
        return (im1, im2, label)


class RandomAffine:
    """
    Affine transform an image with random configurations.
    Args:
        size (tuple, optional): The target size after affine transformation. Default: (224, 224).
        translation_offset (float, optional): The maximum translation offset. Default: 0.
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        im_padding_value (float, optional): The padding value of raw image. Default: (128, 128, 128).
        label_padding_value (int, optional): The padding value of annotation image. Default: (255, 255, 255).
    """
    def __init__(self,
                 size=(224, 224),
                 translation_offset=0,
                 max_rotation=15,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 im_padding_value=(128, 128, 128),
                 label_padding_value=(255, 255, 255)):
        self.size = size
        self.translation_offset = translation_offset
        self.max_rotation = max_rotation
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im1, im2, label=None):
        w, h = self.size
        bbox = [0, 0, im1.shape[1] - 1, im1.shape[0] - 1]
        x_offset = (random.random() - 0.5) * 2 * self.translation_offset
        y_offset = (random.random() - 0.5) * 2 * self.translation_offset
        dx = (w - (bbox[2] + bbox[0])) / 2.0
        dy = (h - (bbox[3] + bbox[1])) / 2.0
        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])
        angle = random.random() * 2 * self.max_rotation - self.max_rotation
        scale = random.random() * (self.max_scale_factor - self.min_scale_factor
                                   ) + self.min_scale_factor
        scale *= np.mean(
            [float(w) / (bbox[2] - bbox[0]),
             float(h) / (bbox[3] - bbox[1])])
        alpha = scale * math.cos(angle / 180.0 * math.pi)
        beta = scale * math.sin(angle / 180.0 * math.pi)
        centerx = w / 2.0 + x_offset
        centery = h / 2.0 + y_offset
        matrix = np.array(
            [[alpha, beta, (1 - alpha) * centerx - beta * centery],
             [-beta, alpha, beta * centerx + (1 - alpha) * centery],
             [0, 0, 1.0]])
        matrix = matrix.dot(matrix_trans)[0:2, :]
        im1 = cv2.warpAffine(
            np.uint8(im1),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.im_padding_value)
        if im2 is not None:
            im2 = cv2.warpAffine(
                np.uint8(im2),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
        if label is not None:
            label = cv2.warpAffine(
                np.uint8(label),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT)
        return (im1, im2, label)