import os
import logging
import cv2
from PIL import Image

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from cellmaps_utils import constants

opj = os.path.join
ope = os.path.exists


logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    def __init__(
        self, image_dir, outdir, image_size=512, crop_size=0, in_channels=4, suffix=".jpg",
        alt_image_ids=None,
    ):
        self.image_dir = image_dir
        self.outdir = outdir
        self.suffix = suffix

        self.transform = None

        self.image_size = image_size
        self.crop_size = crop_size
        self.in_channels = in_channels
        if in_channels == 3:
            self.colors = [constants.RED, constants.GREEN, constants.BLUE]
        elif in_channels == 4:
            self.colors = [constants.RED, constants.GREEN, constants.BLUE,
                           constants.YELLOW]
        else:
            raise ValueError(in_channels)
        self.random_crop = False

        if alt_image_ids is not None:

            self.image_ids = alt_image_ids
            logger.debug('setting alt image ids: ' + str(self.image_ids))
        else:
            # should we get image names from all color directories
            # and then let sort uniq do its work or do we assume we are good?
            image_names = os.listdir(os.path.join(self.image_dir, 'red'))
            logger.debug('Found: ' + str(len(image_names)) +
                         ' images in red directory')

            # eg. ffd91122-bad0-11e8-b2b8-ac1f6b6435d0_red.png -> ffd91122-bad0-11e8-b2b8-ac1f6b6435d0
            self.image_ids = np.sort(
                np.unique(
                    [image_name[: image_name.rfind("_")] for image_name in image_names if image_name.endswith(self.suffix)]
                )
            )
        self.num = len(self.image_ids)
        logger.debug('Found ' + str(self.num) + ' unique image_ids')
        if self.num > 0:
            logger.debug('First image_id: ' + str(self.image_ids[0]))

    def set_transform(self, transform=None):
        self.transform = transform

    def set_random_crop(self, random_crop=False):
        self.random_crop = random_crop

    def crop_image(self, image):
        random_crop_size = int(np.random.uniform(self.crop_size, self.image_size))
        x = int(np.random.uniform(0, self.image_size - random_crop_size))
        y = int(np.random.uniform(0, self.image_size - random_crop_size))
        crop_image = image[x : x + random_crop_size, y : y + random_crop_size]
        return crop_image

    def read_rgby(self, image_id):
        # resize image
        for color in self.colors:
            try:

                image = np.array(Image.open(
                    opj(self.image_dir, color, "%s_%s%s" % (image_id, color, self.suffix))))[:, :, constants.COLOR_INDEXS.get(color)]

            except Exception as e:
                # for issue #12 added proper debug statement instead of just saying bad image
                logger.debug('Caught exception loading image : ' +
                             str(image_id) + ' for color ' +
                             str(color) +
                             ' using PIL : ' +
                             str(e) + ' going to try cv2')
                image = cv2.imread(opj(self.image_dir, color, "%s_%s%s" % (image_id, color, self.suffix)))[:, :, -1::-1][:, :, constants.COLOR_INDEXS.get(color)]
            
            h, w = image.shape[:2]
            if h != self.image_size or w != self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(opj(self.outdir, color + '_resize', "%s_%s%s" % (image_id, color, self.suffix)), image,  [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
        image = [
            cv2.imread(
                opj(self.outdir, color + '_resize', "%s_%s%s" % (image_id, color, self.suffix)),
                cv2.IMREAD_GRAYSCALE,
            )
            for color in self.colors
        ]

        if image[0] is None:
            logger.debug(str(self.image_dir) + ' ' + str(image_id))
        
        image = np.stack(image, axis=-1)
        logger.info(str(image.shape))
        h, w = image.shape[:2]
        if self.image_size != h or self.image_size != w:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.random_crop and self.crop_size > 0:
            image = self.crop_image(image)
        if self.crop_size > 0:
            h, w = image.shape[:2]
            if self.crop_size != h or self.crop_size != w:
                image = cv2.resize(
                    image,
                    (self.crop_size, self.crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )

        return image

    def image_to_tensor(self, image, mean=0, std=1.0):
        image = image.astype(np.float32)
        image = (image - mean) / std
        image = image.transpose((2, 0, 1))
        tensor = torch.from_numpy(image)
        return tensor

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image = self.read_rgby(image_id)

        if self.transform is not None:
            image = self.transform(image)

        image = image / 255.0
        image = self.image_to_tensor(image)

        return image, index

    def __len__(self):
        return self.num


def augment_default(image):
    return image


def augment_flipud(image):
    image = np.flipud(image)
    return image


def augment_fliplr(image):
    image = np.fliplr(image)
    return image


def augment_transpose(image):
    image = np.transpose(image, (1, 0, 2))
    return image


def augment_flipud_lr(image):
    image = np.flipud(image)
    image = np.fliplr(image)
    return image


def augment_flipud_transpose(image):
    image = augment_flipud(image)
    image = augment_transpose(image)
    return image


def augment_fliplr_transpose(image):
    image = augment_fliplr(image)
    image = augment_transpose(image)
    return image


def augment_flipud_lr_transpose(image):
    image = augment_flipud(image)
    image = augment_fliplr(image)
    image = augment_transpose(image)
    return image
