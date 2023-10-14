from monai.transforms import MapTransform


# Define a Monai transform that will change mask value 1 to 2 if 'label' is 1, and keep mask value 1 if 'label' is 0
class EncodeClassIntoMask(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if(d['label'] == 1):
                # Set tensor d[key] to 2 where its 1
                d[key][d[key] == 1] = 2
            
        return d