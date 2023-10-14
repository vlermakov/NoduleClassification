import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import pandas as pd
import generaltools as gt
import trainingblock as tb
import pydicom as dicom
import glob
import os
import skimage.measure as skm

# Class for loading and processing the CT scan data for LUNA 2016 dataset
class CTScan:
    def __init__(self):
        self.scanID = 0
        self.ct_scan = None
        self.ct_scan_labels = None
        self.ct_scan_nodule_labels = None
        self.detected_nodule_mask = None
        self.mask_labeled = None

        self.origin = []
        self.spacing = []
        self.orignal_spacing = []
    

    def ExtractLungMask(self):
        self.ct_scan_labels = gt.segment_lung_mask(self.ct_scan)

    def ExtractSliceGrid(self):
        gridSize = self.ct_scan.shape

        blocks = []
        
        for z in range(0,gridSize[0]-tb.TRAINING_BLOCK_Z,tb.HALF_TRAINING_BLOCK_Z):
            for y in range(0,gridSize[1]-tb.TRAINING_BLOCK_Y,tb.HALF_TRAINING_BLOCK_Y):
                for x in range(0,gridSize[2]-tb.TRAINING_BLOCK_X,tb.HALF_TRAINING_BLOCK_X):

                    # If lung masks exists, check if the overlap with the lung is significant and only look at relevant cubes
                    if(self.ct_scan_labels is not None):
                        if(np.sum(self.ct_scan_labels[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X] >= 1) < 0.25*tb.TRAINING_BLOCK_VOLUME):
                            continue

                    data = self.ct_scan[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]
                    mask = self.ct_scan_nodule_labels[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]

                    blocks.append(tb.SimpleBlock(data = data,mask = mask,origin = (z,y,x)))

        return blocks


    def LoadMaskFromSliceGrid(self,blocks):
        self.mask = np.zeros(self.ct_scan.shape).astype(np.bool_)

        for block in blocks:
            self.mask[block.origin[0]:block.origin[0]+tb.TRAINING_BLOCK_Z,block.origin[1]:block.origin[1]+tb.TRAINING_BLOCK_Y,block.origin[2]:block.origin[2]+tb.TRAINING_BLOCK_X] = block.mask


    def PlotIntensityHistogram(self):
        df = pd.DataFrame({'color': self.ct_scan[:, :, :].flatten()})
        df['color'].plot(kind='hist', bins=50, title='Intensity Histogram')

    def PlotLabelHistogram(self):
        df2 = pd.DataFrame({'color': self.ct_scan_labels[:, :, :].flatten()})
        df2['color'].plot(kind='hist', bins=50, title='Label Histogram')

    def PlotSlice(self, slice_number):
        fig2 = plt.figure()

        axial=plt.subplot(1,1,1)
        axial.set_title('Original Image')
        plt.imshow(self.ct_scan[slice_number, :, :], cmap=plt.cm.gray)
        plt.imshow(self.ct_scan_labels[slice_number, :, :], alpha=0.5)
        if(self.ct_scan_nodule_labels is not None):
            plt.imshow(self.ct_scan_nodule_labels[slice_number, :, :], alpha=0.5)
        plt.show()

    def AssembleSliceGrid(self, slices):

        new_mask = np.zeros(self.ct_scan.shape).astype(np.bool_)

        # Iterate through all the extracted slices, mapping them to an empty mask based on their origin
        for slice in slices:
            new_mask[slice.origin[0]:slice.origin[0]+tb.TRAINING_BLOCK_Z,slice.origin[1]:slice.origin[1]+tb.TRAINING_BLOCK_Y,slice.origin[2]:slice.origin[2]+tb.TRAINING_BLOCK_X] = np.rollaxis(slice.mask.squeeze(),2,0)

        self.detected_nodule_mask = new_mask

    def MarkNodule(self, nodule):
        # If nodule mask doesn't exist, create it
        if(self.ct_scan_nodule_labels is None):
            self.ct_scan_nodule_labels = np.zeros(self.ct_scan.shape).astype(np.bool_)

        # nodule bounding box in voxel space
        wldTopCorner = (nodule["coordX"]+nodule["diameter_mm"]/2,nodule["coordY"]+nodule["diameter_mm"]/2,nodule["coordZ"]+nodule["diameter_mm"]/2)
        wldBottomCorner = (nodule["coordX"]-nodule["diameter_mm"]/2,nodule["coordY"]-nodule["diameter_mm"]/2,nodule["coordZ"]-nodule["diameter_mm"]/2)
        
        voxTopCorner = (np.ceil(gt.world_2_voxel(wldTopCorner, self.origin, self.spacing))).astype(int)
        voxBottomCorner = gt.world_2_voxel(wldBottomCorner, self.origin, self.spacing).astype(int)
        
        for x in range(int(voxBottomCorner[0]),int(voxTopCorner[0])):
            for y in range(int(voxBottomCorner[1]),int(voxTopCorner[1])):
                for z in range(int(voxBottomCorner[2]),int(voxTopCorner[2])):
                    # if distance between voxel and nodule center is less than nodule diameter/2
                    # mark the nodule in the nodule mask
                    if np.linalg.norm(gt.voxel_2_world((x,y,z),self.origin,self.spacing)-np.array([nodule["coordX"],nodule["coordY"],nodule["coordZ"]])) < nodule["diameter_mm"]/2:
                        self.ct_scan_nodule_labels[z,y,x] = 1
                        
    # This can be useful to generate a histogram of the nodules.
    def ExtractNoduleIntensity(self):
        if(self.ct_scan_nodule_labels is None):
            return np.array([])
        else:
            # return values of all voxels in ct_scan  that are marked as nodules
            return self.ct_scan[self.ct_scan_nodule_labels == 1].flatten()

    # This can be useful to generate a histogram of the scan.
    def ExtractScanIntensity(self):
        return self.ct_scan.flatten()
    
    # Use skimage.measure to extract the nodules from the mask and find their centers
    def getNodulesFromMask(self):
        mask = self.ct_scan_nodule_labels > 0

        self.mask_labeled = skm.label(mask)

        return self.annotations.getNoduleCentersFromMask(self.mask_labeled,self.ct_scan_nodule_labels, self.origin, self.spacing)

    # Normalize the scan to be between -1 and 1
    # Extreme data thats below -1000 or above 1000 is set to -1000 or 1000
    # This is ok because the data is in Hounsfield units
    # and we know that air is -1000 and bone is 1000
    def NormalizeScan(self):
        self.ct_scan = gt.normalize_scan(self.ct_scan)

    # Cut out tb.TRAINING_BLOCK_SIZExtb.TRAINING_BLOCK_SIZExtb.TRAINING_BLOCK_SIZE block of voxels from ct_scan around nodule center of the nodule
    def GenerateTrainingBlock(self, present_nodules, nodule_center, isWorldCoord = False):
        # nodule bounding box in voxel space
        if(isWorldCoord):
            voxCenerOfNodule = gt.world_2_voxel((nodule_center["coordX"],nodule_center["coordY"],nodule_center["coordZ"]), self.origin, self.spacing).astype(int)
        else:
            voxCenerOfNodule = (int(nodule_center["coordX"]),int(nodule_center["coordY"]),int(nodule_center["coordZ"]) ) 
        
        # Cut out tb.TRAINING_BLOCK_SIZExtb.TRAINING_BLOCK_SIZExtb.TRAINING_BLOCK_SIZE cube of voxels from ct_scan around nodule center of the nodule
        # and normalize the cube
        block_min_x, block_max_x = np.max([voxCenerOfNodule[0]-tb.HALF_TRAINING_BLOCK_X,0]), np.min([voxCenerOfNodule[0]+tb.HALF_TRAINING_BLOCK_X, self.ct_scan.shape[2]])
        block_min_y, block_max_y = np.max([voxCenerOfNodule[1]-tb.HALF_TRAINING_BLOCK_Y,0]), np.min([voxCenerOfNodule[1]+tb.HALF_TRAINING_BLOCK_Y, self.ct_scan.shape[1]])
        block_min_z, block_max_z = np.max([voxCenerOfNodule[2]-tb.HALF_TRAINING_BLOCK_Z,0]), np.min([voxCenerOfNodule[2]+tb.HALF_TRAINING_BLOCK_Z, self.ct_scan.shape[0]])

        block = np.full(tb.TRAINING_BLOCK_SIZE,-1.0,dtype=np.float32)
        mask = np.zeros(tb.TRAINING_BLOCK_SIZE,dtype=np.int8)
        lung_mask = np.zeros(tb.TRAINING_BLOCK_SIZE,dtype=np.bool_)
        nodule_index_mask = np.zeros(tb.TRAINING_BLOCK_SIZE,dtype=np.int32)

        target_block_min_x, target_block_max_x = tb.HALF_TRAINING_BLOCK_X - (voxCenerOfNodule[0]-block_min_x), tb.HALF_TRAINING_BLOCK_X + (block_max_x-voxCenerOfNodule[0])
        target_block_min_y, target_block_max_y = tb.HALF_TRAINING_BLOCK_Y - (voxCenerOfNodule[1]-block_min_y), tb.HALF_TRAINING_BLOCK_Y + (block_max_y-voxCenerOfNodule[1])
        target_block_min_z, target_block_max_z = tb.HALF_TRAINING_BLOCK_Z - (voxCenerOfNodule[2]-block_min_z), tb.HALF_TRAINING_BLOCK_Z + (block_max_z-voxCenerOfNodule[2])
        
        
        block[target_block_min_z:target_block_max_z,target_block_min_y:target_block_max_y,target_block_min_x:target_block_max_x] = self.ct_scan[block_min_z:block_max_z,block_min_y: block_max_y,block_min_x: block_max_x]
        mask[target_block_min_z:target_block_max_z,target_block_min_y:target_block_max_y,target_block_min_x:target_block_max_x] = self.ct_scan_nodule_labels[block_min_z:block_max_z,block_min_y: block_max_y,block_min_x: block_max_x]
        lung_mask[target_block_min_z:target_block_max_z,target_block_min_y:target_block_max_y,target_block_min_x:target_block_max_x] = (self.ct_scan_labels[block_min_z:block_max_z,block_min_y: block_max_y,block_min_x: block_max_x] >= 1).astype(np.bool_)
        nodule_index_mask[target_block_min_z:target_block_max_z,target_block_min_y:target_block_max_y,target_block_min_x:target_block_max_x] = self.mask_labeled[block_min_z:block_max_z,block_min_y: block_max_y,block_min_x: block_max_x]
        
        # From the nodule_index_mask get a list of unique nodule indexes
        # if the nodule_index_mask is empty, nodule_index will be 0
        nodule_indices = np.unique(nodule_index_mask)[1:]

        # If there are multiple nodules, check which of the nodule indices is the biggest diameter, and use that
        if(len(nodule_indices) > 0):
            nodule_index = nodule_indices[np.argmax([present_nodules.iloc[nodule_index-1].diameter_mm for nodule_index in nodule_indices])]
        else:
            nodule_index = np.max(nodule_index_mask)

        label = np.zeros(11,dtype=np.float32)

        if(nodule_index > 0):
            label[0] = 1
            label[1] = present_nodules.iloc[nodule_index-1].diameter_mm
            label[2] =  present_nodules.iloc[nodule_index-1,0]
        
        block_location = (voxCenerOfNodule[2]-tb.HALF_TRAINING_BLOCK_Z, 
                 voxCenerOfNodule[1]-tb.HALF_TRAINING_BLOCK_Y, 
                 voxCenerOfNodule[0]-tb.HALF_TRAINING_BLOCK_X)

        return tb.TrainingBlock(
            block, 
            np.array(mask>0,dtype=np.bool_), 
            label, 
            lung_mask, 
            self.scanID,block_location)

    # Starting from Z = 0, sliding by 64 voxels, cut out 128x128x128 cube of voxels from ct_scan
    # cut out all the cubes at that Z level iterating through X and Y axis, then move on to the next z level by 64 voxels
    # if the corresponding cube in ct_scan_nodule_labels contains voxels that are != 0, label the cube as 1
    # else label the cube as 0
    def GenerateAllTrainingBlocks(self, present_nodules, drop_percentage, FRACTION_OF_LUNG_IN_CUBE = 25):
        training_blocks = []
        
        for z in range(0,self.ct_scan.shape[0]-tb.TRAINING_BLOCK_Z,tb.HALF_TRAINING_BLOCK_Z):
            for y in range(0,self.ct_scan.shape[1]-tb.TRAINING_BLOCK_Y,tb.HALF_TRAINING_BLOCK_Y):
                for x in range(0,self.ct_scan.shape[2]-tb.TRAINING_BLOCK_X,tb.HALF_TRAINING_BLOCK_X):
                    cube = self.ct_scan[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]
                    mask = self.ct_scan_nodule_labels[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]
                    if(cube.shape == tb.TRAINING_BLOCK_SIZE):

                        # Discard cubes that have nodules on border (want to make sure nodules are completely included in the cube for training)
                        if(mask[0,:,:].sum() + mask[-1,:,:].sum() + mask[:,0,:].sum() + mask[:,-1,:].sum() + mask[:,:,0].sum() + mask[:,:,-1].sum() > 0):
                            continue

                        # Check how much of the cube overlaps with the lung
                        # if the cube overlaps with the lung less than 25%, discard the cube
                        lung_mask = (self.ct_scan_labels[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X] >= 1).astype(np.bool_)
                        
                        # Discard cubes that don't contain a significant portion of the lung
                        if(np.sum(lung_mask) < tb.TRAINING_BLOCK_VOLUME * (FRACTION_OF_LUNG_IN_CUBE / 100.0)):
                            #print("Less than {}% overlap with lung, discarding cube".format(100*(FRACTION_OF_LUNG_IN_CUBE/100)))
                            continue


                        # only keep a small portion of random negative cubes to balance the data
                        if(np.max(self.ct_scan_nodule_labels[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]) == 0):
                            if(np.random.randint(0,100,) > drop_percentage):
                                continue
                        
                        if(self.mask_labeled is None):
                            nodule_index_mask = -1
                        else:
                            nodule_index_mask = self.mask_labeled[z:z+tb.TRAINING_BLOCK_Z,y:y+tb.TRAINING_BLOCK_Y,x:x+tb.TRAINING_BLOCK_X]

                        nodule_index = np.max(nodule_index_mask)
                        label = np.zeros(11,dtype=np.float32)

                        if(nodule_index > 0):
                            label[0] = 1
                            label[1] = present_nodules.iloc[nodule_index-1].diameter_mm
                            label[2:] =  present_nodules.iloc[nodule_index-1,0:9]

                        block_location = (z, y, x)

                        training_blocks.append(
                            tb.TrainingBlock(
                                cube, 
                                np.array(mask>0,dtype=np.bool_), 
                                label, 
                                lung_mask, 
                                self.scanID, 
                                block_location))
                        
        return np.array(training_blocks)

    def GenerateTrainingSlices(self, nodule):
        # nodule bounding box in voxel space
        wldTopCorner = (nodule["coordX"]+nodule["diameter_mm"]/2,nodule["coordY"]+nodule["diameter_mm"]/2,nodule["coordZ"]+nodule["diameter_mm"]/2)
        wldBottomCorner = (nodule["coordX"]-nodule["diameter_mm"]/2,nodule["coordY"]-nodule["diameter_mm"]/2,nodule["coordZ"]-nodule["diameter_mm"]/2)
        
        voxTopCorner = (np.ceil(gt.world_2_voxel(wldTopCorner, self.origin, self.spacing))).astype(int)
        voxBottomCorner = gt.world_2_voxel(wldBottomCorner, self.origin, self.spacing).astype(int)

        # center of the nodule in voxel space
        voxCenter = (np.ceil(gt.world_2_voxel((nodule["coordX"],nodule["coordY"],nodule["coordZ"]), self.origin, self.spacing))).astype(int)

        # height of the nodule in voxels
        noduleHeight = voxTopCorner[2] - voxBottomCorner[2]

        # if nodule is larger than 3 voxels in height, take only 3 center slices
        if(noduleHeight > 3):
            # take only 3 center slices of the nodule 
            voxBottomCorner[2] = voxCenter[2] + 1
            voxTopCorner[2] = voxCenter[2] - 1
        
        # Extract the slices
        slices = []
        masks = []

        for z in range(int(voxTopCorner[2]),int(voxBottomCorner[2])):
            slices.append(self.ct_scan[z, :, :])
            masks.append(self.ct_scan_nodule_labels[z, :, :])
            
        return slices,masks
    # Replace all regions that aren't part of a lung with -1000
    def RemoveNonLungRegions(self):
        # For all voxels that are not part of the lungs, set them to -1000
        self.ct_scan[self.ct_scan_labels == 0] = -1000


class LunaScan(CTScan):

    def __init__(self, scanDirectory, scanID,annotations = None, lungLabelsDirectory = None):
        super().__init__()

        if lungLabelsDirectory is not None:
            filename_segments = lungLabelsDirectory+scanID +".mhd"
        filename_image = scanDirectory+scanID+".mhd"

        self.scanID = scanID
        
        # Load the data once
        try:
            print ("Loading image file: ",filename_image)
            self.ct_scan, self.origin, self.spacing = self.load_itk(filename_image)
            if lungLabelsDirectory is not None:
                print ("Loading label file: ",filename_segments)
                self.ct_scan_labels, origin_labels, spacing_labels = self.load_itk(filename_segments)
                self.ct_scan_labels = (self.ct_scan_labels > 1).astype(np.int8)
        except:
            raise IOError("Error: Could not load image file")


        print("Loaded files.")

        # Check if the origin and spacing of the image and labels match
        if(self.ct_scan_labels is not None and ((abs(np.subtract(self.origin,origin_labels)) > 1).any() or (abs(np.subtract(self.spacing,spacing_labels)) > 1).any())):
            raise ValueError("Error: Origin or spacing of image and labels do not match")

        self.orignal_spacing = self.spacing

        # Import the annotations and project them to the CTScan nodule mask
        if(annotations):
            try:
                self.annotations = annotations
                self.ct_scan_nodule_labels = np.zeros(self.ct_scan.shape).astype(np.int8)
                
                self.ct_scan_nodule_labels = self.annotations.importAnnotations(scanID, self.ct_scan_nodule_labels, self.origin, self.spacing)

            except:
                raise IOError("Error: Could not load LIDC annotations")
            
        # # Resample the image and labels to 1mm isotropic voxels
        self.ct_scan, self.spacing = gt.resample_new(self.ct_scan,self.orignal_spacing,[1,1,1])

        if(self.ct_scan_nodule_labels is not None):
            self.ct_scan_nodule_labels, self.spacing = gt.resample_new(self.ct_scan_nodule_labels,self.orignal_spacing,[1,1,1],True)

        # # Resample lung labels to 1mm isotropic voxels
        if self.ct_scan_labels is not None:
            self.ct_scan_labels, self.spacing = gt.resample_new(self.ct_scan_labels,self.orignal_spacing,[1,1,1],True)

    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, 
    origin and spacing of the image.
    '''
    def load_itk(self, filename):
        #print("load_itk: filename = ", filename)

        # Reads the image using SimpleITK
        # indexes are z,y,x (notice the ordering)
        itkimage = sitk.ReadImage(filename) 
        
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage).astype(np.float32)
        
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(itkimage.GetOrigin())
        
        # Read the spacing along each dimension
        spacing = np.array(itkimage.GetSpacing())

        return ct_scan, origin, spacing


    '''
    This function is used to save the image array to a '.mhd' file using SimpleITK.
    '''
    def save_itk(self, image, filename, origin, spacing):
        itkimage = sitk.GetImageFromArray(image)
        itkimage.SetOrigin(origin)
        itkimage.SetSpacing(spacing)
        sitk.WriteImage(itkimage, filename)


    
class DICOMScan(CTScan):
    def __init__(self, scanDirectory, annotations = None):        
        super().__init__()

        # Check if scan directory exists
        if(not os.path.isdir(scanDirectory)):
            raise IOError("Error: Scan directory does not exist")

        # Get all DICOM files in this folder but not subfolders
        dicom_files = glob.glob(scanDirectory + "/*")


        dicom_files = [file for file in dicom_files if os.path.isfile(file)]

        # Check if there are any DICOM files in the directory
        if(len(dicom_files) == 0):
            raise IOError("Error: No DICOM files in directory")


        slices = []
        for file_name in dicom_files:
            ds =  dicom.dcmread(file_name,force=True)
            if(hasattr(ds,"ImagePositionPatient")):
                slices.append(ds)
        
        self.scanID = slices[0].PatientID

        #sort slices by ImagePositionPatient
        slices = sorted(slices,key=lambda x:x.ImagePositionPatient[2])

        pixel_spacing = slices[0].PixelSpacing
        slices_thickess = slices[0].SliceThickness if hasattr(slices[0],'SliceThickness') else abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        self.origin = np.array([slices[0].ImagePositionPatient[0],slices[0].ImagePositionPatient[1],slices[0].ImagePositionPatient[2]]).astype(np.float32)
        self.spacing = np.array([pixel_spacing[0],pixel_spacing[1],slices_thickess])

        img_shape = list(slices[0].pixel_array.shape)
        img_shape.insert(0,len(slices))
        self.ct_scan=np.zeros(img_shape, dtype=np.float32)

        for i,s in enumerate(slices):
            array2D=s.pixel_array*ds.RescaleSlope+ds.RescaleIntercept
            self.ct_scan[i,:,:]= array2D

        self.orignal_spacing = self.spacing

        # Import the annotations and project them to the CTScan nodule mask
        if(annotations):
            try:
                self.annotations = annotations
                self.ct_scan_nodule_labels = np.zeros(self.ct_scan.shape).astype(np.int8)
                
                self.ct_scan_nodule_labels = self.annotations.importAnnotations(self.scanID, self.ct_scan_nodule_labels, self.origin, self.spacing)

            except:
                raise IOError("Error: Could not load annotations")
            

        # Resample the image and labels to 1mm isotropic voxels
        self.ct_scan, self.spacing = gt.resample_new(self.ct_scan,self.orignal_spacing,[1,1,1])
        
        if(self.ct_scan_nodule_labels is not None):
            self.ct_scan_nodule_labels, self.spacing = gt.resample_new(self.ct_scan_nodule_labels,self.orignal_spacing,[1,1,1],True)
        else:
            self.ct_scan_nodule_labels = np.zeros(self.ct_scan.shape).astype(np.int8)
