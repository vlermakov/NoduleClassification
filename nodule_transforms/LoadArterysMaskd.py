
from monai.transforms import MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix
from monai.utils import ensure_tuple, ensure_tuple_rep


from monai.data import MetaTensor


DEFAULT_POST_FIX = PostFix.meta()

import numpy as np

import torch

from annotations import ArterysAnnotations

class LoadArterysMaskd(MapTransform):

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

            annotations = ArterysAnnotations(d[key])

            # Rearrange the mask to be z,y,x  (instead of x,y,z)
            mask = np.moveaxis(mask, [0,1,2], [2,1,0])
            
            data = annotations.importAnnotations(d['patch_id'], mask,d['image_meta_dict']['spacing'],d['image_meta_dict']['original_affine'][:,-1][:-1])

            # Rearrange the data to be x,y,z  (instead of z,y,x)
            data = np.moveaxis(data, [0,1,2], [2,1,0])

            #if self._loader.image_only:
            # add first channel to mask annd convert to torch tensor
            d[key] = MetaTensor(torch.from_numpy(data).unsqueeze(0), meta=d['image'].meta.copy()).type(torch.float32)

            d[key + "_meta_dict"] = d['image_meta_dict'].copy()

            nodules_df = annotations.getAnnotations()

            # Find location of the id nodule in the mask and save it with meta data
            nodule_center_position = torch.tensor(nodules_df[nodules_df['id'] == d['patch_id']]['position'].values[0], dtype=torch.float32)
            nodule_center_offset = torch.tensor(nodules_df[nodules_df['id'] == d['patch_id']]['centroid'].values[0], dtype=torch.float32)
            nodule_center = nodule_center_position + nodule_center_offset
            
            d[key + "_meta_dict"]["nodule_centroid_voxel"] = nodule_center
            
        return d