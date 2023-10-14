
from monai.transforms import (
    SpatialCrop
)
from monai.transforms import (
    MapTransform,
    BorderPad
)
from monai.config import KeysCollection

import numpy as np

import torch


class CropToNoduled(MapTransform):
    def __init__(self, keys: KeysCollection, meta_keys: KeysCollection | None = None, *args, **kwargs) -> None:
        super().__init__(keys, meta_keys, *args, **kwargs)

    def __call__(self, data):
        nodule_centroid_original = data['mask_meta_dict']['nodule_centroid_voxel'].clone().detach()

        # add extra 1 to the end of the nodule centroid to make it a 4x1 vector
        nodule_centroid_original = torch.cat((nodule_centroid_original, torch.tensor([1.0])))

        # Transform the nodule centroid based on the affine transforms applied to the mask
        #nodule_centroid_transformed = torch.linalg.inv(torch.tensor(torch.abs(data['mask'].affine[0:3,0:3]),dtype=torch.float32)) @ torch.from_numpy(np.abs(data['mask'].meta['original_affine'][0:3,0:3]).astype(np.float32)) @ nodule_centroid_original
        reverse_affine = torch.abs(data['mask'].affine.type(torch.float32))
        nodule_centroid_transformed = (torch.linalg.inv(reverse_affine) @ torch.from_numpy(np.abs(data['mask'].meta['original_affine']).astype(np.float32)) @ nodule_centroid_original)[0:3]

        padder = BorderPad(96, mode='constant')
        
        # Add 96 to all dimensions of nodule_centroid_transformed
        nodule_centroid_transformed = nodule_centroid_transformed + 96

        cropper = SpatialCrop(roi_center=nodule_centroid_transformed, roi_size=(96,96,96))

        for key in self.key_iterator(data):
            data[key] = cropper(padder(data[key]))

        return data
