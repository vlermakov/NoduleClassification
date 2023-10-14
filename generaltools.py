import numpy as np
import numpy as np
from skimage import morphology
from skimage import measure
from skimage import segmentation
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import os
import scipy.ndimage
from matplotlib import pyplot as plt
import glob
from skimage.transform import resize


def ClearTrainingDataDirectory(folder = './training_data'):
    # Remove all files in the training data directory
    files = glob.glob(folder + "/cubes/*")
    for f in files:
        os.remove(f)
    files = glob.glob(folder + "/labels/*")
    for f in files:
        os.remove(f)
    
    files = glob.glob(folder + "/cubemasks/*")
    for f in files:
        os.remove(f)
        
    files = glob.glob(folder + "/lungmasks/*")
    for f in files:
        os.remove(f)

    files = glob.glob(folder + "/*.hd5")
    for f in files:
        os.remove(f)


'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def normalize_scan(scan):
    # For all voxels that have intensity below -1000, set them to -1000
    scan[scan < -1000] = -1000
    # For all voxels that have intensity above 1000, set them to 1000
    scan[scan > 1000] = 1000
    
    scan = scan / 1000.0
    
    return scan


# Resample the scan to a new spacing (ex. 1mm x 1mm x 1mm voxels)
# This is much faster than the resample function below
# Got this idea from nnUnet
def resample_new(original, spacing, new_spacing=[1,1,1],isMask = False):
    resize_factor = spacing/new_spacing

    # resize factor taking into account that z axis is first in the scan matrix
    new_real_shape = original.shape * np.array([resize_factor[2],resize_factor[0],resize_factor[1]])

    new_shape = np.round(new_real_shape)

    if(isMask):
        # Force nearest neighbor interpolation for masks
        image = resize(original, output_shape=new_shape, mode="edge", anti_aliasing=False,order=0)
    else:
        image = resize(original, output_shape=new_shape, mode="edge", anti_aliasing=False)

    return image, np.array(new_spacing)

# Resample the scan to a new spacing (ex. 1mm x 1mm x 1mm voxels)
def resample(original, spacing, new_spacing=[1,1,1]):
    resize_factor = spacing/new_spacing

    # resize factor taking into account that z axis is first in the scan matrix
    new_real_shape = original.shape * np.array([resize_factor[2],resize_factor[0],resize_factor[1]])

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / original.shape

    new_spacing = np.array([spacing[2]/real_resize_factor[0],spacing[0]/real_resize_factor[1],spacing[1]/real_resize_factor[2]])
    
    image = scipy.ndimage.interpolation.zoom(original, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    
    dilated = morphology.dilation(binary_image,np.ones([4,4,4]))
    binary_image = morphology.erosion(dilated,np.ones([5,5,5]))

    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    
    # Do this for every corner of the 3D image labels
    for corner in [(0,0,0), (0,0,image.shape[2]-1), (0,image.shape[1]-1,0), (0,image.shape[1]-1,image.shape[2]-1), (image.shape[0]-1,0,0), (image.shape[0]-1,0,image.shape[2]-1), (image.shape[0]-1,image.shape[1]-1,0), (image.shape[0]-1,image.shape[1]-1,image.shape[2]-1)]:
        if(binary_image[corner] == 2):
            continue

        background_label = labels[corner]

        #Fill the air around the person
        binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image