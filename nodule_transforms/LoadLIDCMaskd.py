
from monai.transforms import MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix
from monai.utils import ensure_tuple, ensure_tuple_rep


from monai.data import MetaTensor


DEFAULT_POST_FIX = PostFix.meta()

import numpy as np

import torch

from annotations import LIDCAnnotations

import skimage.measure as skm

class LoadLIDCMaskd(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(keys, allow_missing_keys)

        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting
        
        self.image_only = image_only
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep
        self.expanduser = expanduser

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):

            # Create a tensor of size of ['image'] with all zeros
            mask = np.zeros(d['image'].shape[1:], dtype=np.float32)
            try:

                annotations = LIDCAnnotations()

                # Rearrange the mask to be z,y,x  (instead of x,y,z)
                mask = np.moveaxis(mask, [0,1,2], [2,1,0])
                
                data = annotations.importAnnotations(d[key], mask, d['image_meta_dict']['original_affine'][:,-1][:-1], d['image_meta_dict']['spacing'])

                # Rearrange the data to be x,y,z  (instead of z,y,x)
                data = np.moveaxis(data, [0,1,2], [2,1,0])

                #if self._loader.image_only:
                # add first channel to mask annd convert to torch tensor
                d[key] = MetaTensor(torch.from_numpy(data).unsqueeze(0), meta=d['image'].meta.copy()).type(torch.float32)

                d[key + "_meta_dict"] = d['image_meta_dict'].copy()

                nodules_df = annotations.getNoduleCentersFromMask(data, d['image_meta_dict']['original_affine'][:,-1][:-1],d['image_meta_dict']['spacing'])
                # nodules_df = annotations.getNoduleCentersFromMask(data, d['image_meta_dict']['spacing'], d['image_meta_dict']['original_affine'][:,-1][:-1])

                # Only look at large nodules
                nodules_df = nodules_df[nodules_df['isLarge'] == 1]
            except:
                #if self._loader.image_only:
                data = mask
                # add first channel to mask annd convert to torch tensor
                d[key] = MetaTensor(torch.from_numpy(data).unsqueeze(0), meta=d['image'].meta.copy()).type(torch.float32)

                d[key + "_meta_dict"] = d['image_meta_dict'].copy()

                nodules_df = []

            if(len(nodules_df) == 0):
                # If no nodules were found, then just return the data
                # Pick a random voxel to be the centroid. Generate voxel coordinate as a pytorch tensor torch.float32
                d[key + "_meta_dict"]["nodule_centroid_voxel"] = torch.tensor([np.random.randint(0,data.shape[0]),np.random.randint(0,data.shape[1]),np.random.randint(0,data.shape[2])], dtype=torch.float32)
                d['label'] = "A"
            else:
                # Pick the nodule with highest malignancy
                nodules_df = nodules_df.sort_values(by=['malignancy'], ascending=False)
                d[key + "_meta_dict"]["nodule_centroid_voxel"] = torch.tensor([nodules_df['coordX'].values[0],nodules_df['coordY'].values[0],nodules_df['coordZ'].values[0]], dtype=torch.float32)
                d['label'] = nodules_df['malignancy'].values[0]
            
        return d