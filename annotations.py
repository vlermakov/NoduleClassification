import numpy as np
import os

from skimage.draw import polygon, polygon_perimeter, ellipse

import lidc_annotations as inxml

import generaltools as gt

import pandas as pd

import skimage.measure as skm

import json
import dpath

import zlib
import base64


class Annotations:
    def __init__(self):
        self.target_volume = None
        self.origin = None
        self.original_spacing = None

    def importAnnotations(self, scanID, targetVolume, origin, spacing):
        # self.loadCTScan(scanID)
        # self.loadNoduleAnnotationsFromLIDC(scanID)
        self.target_volume = targetVolume
        self.origin = origin
        self.original_spacing = spacing

    def getNoduleCentersFromMask(self, nodules_label_volume, mask_origin, mask_spacing):
        return None
    

class ArterysAnnotations(Annotations):
    def __init__(self, annotation_json):

        self.annotation_json = annotation_json
        
        self.__nodule_labels_by_reading = None
        self.__nodule_annotations = None
        
        return
    
    def getAnnotationsForPatch(self, patch_id):
        with open(self.annotation_json) as f:
            data = json.load(f)
            
            # Search for all occurrences of the key "patches" in the data
            found = dpath.search(data, '**/patches', yielded=True)
            
            # Iterate through all child elements of each "patches"
            for path, patches in found:
                for patch in patches:
                    # If fields are not defined then skip this patch
                    if patch['id'] is None or patch['height'] is None or patch['width'] is None or patch['depth'] is None:
                        continue

                    if(patch['id'] != patch_id):
                        continue

                    # Get the center of the nodule in the target volume
                    x = int(patch["position"][0])
                    y = int(patch["position"][1])
                    z = int(patch["position"][2])

                    # Decode the Base64 string to bytes
                    compressed_data = base64.b64decode(patch["binary"])

                    # Decompress using zlib
                    decompressed_data = zlib.decompress(compressed_data)
                    decompressed_data = np.frombuffer(decompressed_data, dtype=np.uint8)

                    decompressed_data = np.array(decompressed_data > 0, dtype=np.int8)

                    # Reshape the data based on height, width and depth in the patch
                    patch_data = decompressed_data.reshape((patch['depth'],patch['height'],patch['width']))

                    # # Inscribe the patch into the target volume
                    # self.target_volume[z:z+patch['depth'],y:y+patch['height'],x:x+patch['width']] = patch_data

                    # tmp_buffer = np.zeros(self.target_volume.shape,dtype=np.int8)
                    # tmp_buffer[z:z+patch['depth'],y:y+patch['height'],x:x+patch['width']] = patch_data

                    # self.__nodule_labels_by_reading.append(tmp_buffer)

                    # Take all the patch keys and values and then to self.__nodule_annotations dataframe
                    # Use concatenation method to add the new row to the dataframe
                    return pd.DataFrame(pd.json_normalize(patch))

        return None
    
    def importAnnotations(self, scanID, targetVolume, origin, spacing):
        super().importAnnotations(scanID, targetVolume, origin, spacing)

        self.__nodule_annotations = pd.DataFrame()
        self.__nodule_labels_by_reading  = []
        
        # Open the json file and parse it
        with open(self.annotation_json) as f:
            data = json.load(f)
            
            # Search for all occurrences of the key "patches" in the data
            found = dpath.search(data, '**/patches', yielded=True)
            
            # Iterate through all child elements of each "patches"
            for path, patches in found:
                for patch in patches:
                    # If fields are not defined then skip this patch
                    if patch['id'] is None or patch['height'] is None or patch['width'] is None or patch['depth'] is None:
                        continue

                    # Get the center of the nodule in the target volume
                    x = int(patch["position"][0])
                    y = int(patch["position"][1])
                    z = int(patch["position"][2])

                    # Decode the Base64 string to bytes
                    compressed_data = base64.b64decode(patch["binary"])

                    # Decompress using zlib
                    decompressed_data = zlib.decompress(compressed_data)
                    decompressed_data = np.frombuffer(decompressed_data, dtype=np.uint8)

                    decompressed_data = np.array(decompressed_data > 0, dtype=np.int8)

                    # Reshape the data based on height, width and depth in the patch
                    patch_data = decompressed_data.reshape((patch['depth'],patch['height'],patch['width']))

                    # Inscribe the patch into the target volume, setting 1 where 1 is in the patch, making sure to not overwrite existing values
                    self.target_volume[z:z+patch['depth'],y:y+patch['height'],x:x+patch['width']] = np.maximum(self.target_volume[z:z+patch['depth'],y:y+patch['height'],x:x+patch['width']],patch_data)
                    

                    tmp_buffer = np.zeros(self.target_volume.shape,dtype=np.int8)
                    tmp_buffer[z:z+patch['depth'],y:y+patch['height'],x:x+patch['width']] = patch_data

                    self.__nodule_labels_by_reading.append(tmp_buffer)

                    # Take all the patch keys and values and then to self.__nodule_annotations dataframe
                    # Use concatenation method to add the new row to the dataframe
                    self.__nodule_annotations = pd.concat([self.__nodule_annotations,pd.DataFrame(pd.json_normalize(patch))],ignore_index=True)


        self.__nodule_labels_by_reading = np.array(self.__nodule_labels_by_reading)
        return self.target_volume
    
    def getAnnotations(self):
        return self.__nodule_annotations


    def getNoduleCentersFromMask(self, labeled_mask, nodules_label_volume,  mask_origin, mask_spacing):
        
        props = skm.regionprops_table(labeled_mask,nodules_label_volume,properties=['centroid','equivalent_diameter_area'])
        saved_props = props.copy()
        
        props['coordZ'] = np.round(props.pop('centroid-0').astype(float) / mask_spacing[2] + mask_origin[2]).astype(int)
        props['coordY'] = np.round(props.pop('centroid-1').astype(float) / mask_spacing[1] + mask_origin[1]).astype(int)
        props['coordX'] = np.round(props.pop('centroid-2').astype(float) / mask_spacing[0] + mask_origin[0]).astype(int)

        props['diameter_mm'] = props.pop('equivalent_diameter_area') / mask_spacing[0]

        df = pd.DataFrame(props)

        df.insert(0,'volume',0)
        df.insert(1,'id',0)
        
        label = np.zeros(11)

        for index, row in df.iterrows():
            x = np.round((row['coordX'] - mask_origin[0]) / self.original_spacing[0]).astype(int)
            y = np.round((row['coordY'] - mask_origin[1]) / self.original_spacing[1]).astype(int)
            z = np.round((row['coordZ'] - mask_origin[2]) / self.original_spacing[2]).astype(int)

            centers = [[a[0] + b[0], a[1] + b[1], a[2] + b[2]]  for a, b in zip(self._ArterysAnnotations__nodule_annotations.position ,self._ArterysAnnotations__nodule_annotations.centroid)]

            # Find the closest of the annotations to this point
            distance = np.linalg.norm(np.array(centers) - np.array([x,y,z]),axis=1)
            
            # Find the closest annotation
            iReading = np.argmin(distance)
            
            df.at[index,'volume'] = self.__nodule_annotations.iloc[iReading].volume
            df.at[index,'id'] = self.__nodule_annotations.iloc[iReading].id

        # Iterate through the nodules that aren't yet included in the dataframe and check if they 
        # also overlap an existing region (if the same region is shared by several labels)
        unmatched_nodules = self.__nodule_annotations[~self.__nodule_annotations.id.isin(df.id)]

        for index, row in unmatched_nodules.iterrows():
            (x,y,z) = np.round(((row['world_centroid'] - self.origin))).astype(int)


            region_id = labeled_mask[z,y,x]

            # If there is an overlap with the region and region's id doesn't match the nodule id then we should insert an extra row with the same
            # # region information, but the additional patch id and volume information.
            if(region_id > 0 and region_id != row.id):
                df = pd.concat([df,pd.DataFrame([[row.volume,row.id,props['coordZ'][region_id-1],props['coordY'][region_id-1],props['coordX'][region_id-1],props['diameter_mm'][region_id-1]]],columns=df.columns)],ignore_index=True)
                

        return df
                    
class LIDCAnnotations(Annotations):
    def __init__(self):

        self.__nodule_labels_by_reading = None
        self.__nodule_annotations = None
        
        return
    
    def getAnnotations(self):
        return self.__nodule_annotations

    def getNoduleCentersFromMask(self, nodules_label_volume, mask_origin, mask_spacing):
        mask = nodules_label_volume > 0
        
        props = skm.regionprops_table(skm.label(mask),nodules_label_volume,properties=['centroid','equivalent_diameter_area'])
        
        props['coordX'] = props.pop('centroid-0').astype(int)# / mask_spacing[2] + mask_origin[2]
        props['coordY'] = props.pop('centroid-1').astype(int)# / mask_spacing[1] + mask_origin[1]
        props['coordZ'] = props.pop('centroid-2').astype(int)# / mask_spacing[0] + mask_origin[0]

        props['diameter_mm'] = props.pop('equivalent_diameter_area') / mask_spacing[0]

        df = pd.DataFrame(props)

        df.insert(0,'spiculation',0.0)
        df.insert(0,'lobulation',0.0)
        df.insert(0,'margin',0.0)
        df.insert(0,'sphericity',0.0)
        df.insert(0,'calcification',0.0)
        df.insert(0,'internalStructure',0.0)
        df.insert(0,'subtlety',0.0)
        df.insert(0,'malignancy',0.0)
        df.insert(0,'isLarge',0.0)

        label = np.zeros(11)

        for index, row in df.iterrows():
            x = int(row['coordX'])# / self.original_spacing[0])
            y = int(row['coordY'])# / self.original_spacing[1])
            z = int(row['coordZ'])# / self.original_spacing[2])
            
            detectedNodules = self.__nodule_labels_by_reading[:,z,y,x]
            
            largeNoduleMerics = []
            smallNoduleMerics = []

            
            for iReading,code in enumerate(detectedNodules):
                
                if(code >= 10000):
                    iNodule = int(code/10000 - 1)
                    
                    largeNoduleMerics.append([
                        1,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].malignancy,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].subtlety,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].internalStructure,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].calcification,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].sphericity,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].margin,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].lobulation,
                        self.__nodule_annotations.readingSessions[iReading][iNodule].spiculation
                    ])

            if(len(largeNoduleMerics)):
                metrics = np.array(largeNoduleMerics,dtype=np.float32)
                df.iloc[index,0:9] = np.mean(metrics,axis=0)

        return df
    
    def importAnnotations(self, scanID, targetVolume, origin, spacing):
        super().importAnnotations(scanID, targetVolume, origin, spacing)

        # Load LIDC scan annotations from the XML files
        self.__nodule_annotations = inxml.ScanAnnotations(scanID)

        self.__nodule_labels_by_reading = self.__convertNoduleAnnotationsToMaskLayers(self.__nodule_annotations)

        # Combine ct_scan_nodule_labels labels from different reading sessions into a single mask
        return self.__combineReadingSessionNoduleMasks(self.__nodule_labels_by_reading)


    def __combineReadingSessionNoduleMasks(self,__nodule_labels_by_reading):     
        combined_nodule_mask = np.zeros(__nodule_labels_by_reading.shape[1:],dtype=np.int8)

        # Add up counts from all reading sessions
        for iReadingSession in range(inxml.MAX_READINGS_PER_SCAN):
            combined_nodule_mask = combined_nodule_mask + __nodule_labels_by_reading[iReadingSession]

        # If there are more than 1 reading session with a nodule then we should mark it as a nodule
        combined_nodule_mask = (combined_nodule_mask > 1).astype(np.int16)
        
        return combined_nodule_mask
        

    def __convertNoduleAnnotationsToMaskLayers(self, nodule_annotations):
        # Lets create a nodules mask array to convert the annotations to mask
        nodule_labels_by_reading_local = np.zeros((inxml.MAX_READINGS_PER_SCAN,self.target_volume.shape[0],self.target_volume.shape[1],self.target_volume.shape[2])).astype(np.int64)

        for iReadingSession, session in enumerate(nodule_annotations.readingSessions[:inxml.MAX_READINGS_PER_SCAN]):
            for iNodule, nodule in enumerate(session):
                if(isinstance(nodule,inxml.LargeNodule)):
                    for s in nodule.slices:
                        # If there are less than 3 points we can't build a plygon so should skip this slice
                        if(len(s.path) < 3):
                            continue
                        
                        _,_,zVoxel = gt.world_2_voxel((0,0,s.z), self.origin, self.original_spacing)

                        polygon_params = {
                            "r": [x[1] for x in s.path],
                            "c": [x[0] for x in s.path],
                            "shape": (self.target_volume.shape[1],self.target_volume.shape[2])
                        }
                        
                        rr, cc = polygon(**polygon_params)
                        rr2, cc2 = polygon_perimeter(**polygon_params)
                        
                        # Draw polygon and remove premiter since its marked on the outside of the nodule and we only want to mark
                        # pixels that are part of the nodule
                        nodule_labels_by_reading_local[iReadingSession,int(zVoxel),rr,cc] = 10000*(iNodule+1) if s.inclusion else 0

                        if(s.inclusion):
                            nodule_labels_by_reading_local[iReadingSession,int(zVoxel),rr2,cc2] = 0

                    print(f"({iReadingSession}) Large Nodule: { gt.world_2_voxel((0,0,nodule.slices[0].z), self.origin, self.original_spacing)[2] } - { gt.world_2_voxel((0,0,nodule.slices[-1].z), self.origin, self.original_spacing)[2] }")
                elif(isinstance(nodule,inxml.Nodule)):
                    for s in nodule.slices:
                        _,_,zVoxel = gt.world_2_voxel((0,0,s.z), self.origin, self.original_spacing)
                        # get a circle around the nodule
                        rr, cc = ellipse(s.path[0][1], s.path[0][0], 2, 2, (self.target_volume.shape[1],self.target_volume.shape[2]))
                        nodule_labels_by_reading_local[iReadingSession,int(zVoxel),rr,cc] = 100*(iNodule + 1) if s.inclusion else 0
                    
                    pass #print("Small Nodule")
                elif(isinstance(nodule,inxml.NonNodule)):
                    pass

                    # for s in nodule.slices:
                    #     _,_,zVoxel = gt.world_2_voxel((0,0,s.z), self.origin, self.original_spacing)
                    #     # get a circle around the nodule
                    #     rr, cc = ellipse(s.path[0][1], s.path[0][0], 2, 2, (self.ct_scan.shape[1],self.ct_scan.shape[2]))
                    #     self.ct_scan_nodule_labels[int(zVoxel),rr,cc] = 1 if s.inclusion else 0
                    
                    pass #print("Non Nodule")
                else:
                    print("Unknown Nodule")

        return nodule_labels_by_reading_local