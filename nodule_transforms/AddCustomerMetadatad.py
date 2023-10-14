
from monai.transforms import (
    MapTransform
)
from monai.config import DtypeLike, KeysCollection
from monai.utils.enums import PostFix
from monai.utils import ensure_tuple, ensure_tuple_rep


DEFAULT_POST_FIX = PostFix.meta()

import numpy as np

class AddCustomerMetadatad(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data):
        d = dict(data)
        for key, meta_key_postfix in self.key_iterator(d, self.meta_key_postfix):
            for meta_key in self.meta_keys:
                if  meta_key is None:
                    continue
                if meta_key not in d:
                    raise KeyError(f"Metadata key {meta_key} not in data dictionary.")
                d[key].meta[meta_key] = d[meta_key]

        return d
