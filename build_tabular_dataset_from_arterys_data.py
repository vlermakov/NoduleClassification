import numpy as np
import pandas as pd
import multiprocessing as mp
import generaltools as gt
import tqdm
import trainingblock as tb
from trainingtools import processScan
import os
import glob


from annotations import ArterysAnnotations
import ctscan as ct
import json


TRAINING_FOLDER = './cancer_nodules_training'

DATASET_FOLDER = '/media/vlermakov/data/UCSDNodules/'

dfDatasetDefinition = pd.read_csv('dataset_squamous.csv')

VOLUME_TOLERANCE = 4

# Make numpy random predictable
np.random.seed(0)

if __name__ == '__main__':

    folds = ['Metastatic']

    df_process_log = pd.DataFrame(columns=['MRN','Phonetic_ID','Patch_ID','Slice_Thickness','Status'])

    df_patch_features = pd.DataFrame()

    process_parameters = []
    
    for DATASET_FOLD in folds:

        # Read in the dataset definition
        DATASET_SUBFOLDER = os.path.join(DATASET_FOLDER, DATASET_FOLD)
        dataset = pd.read_csv(os.path.join(DATASET_SUBFOLDER, "dataset.csv"), dtype={'MRN': str})

        # Only use part of the dataset where "process" is 1
        dataset = dataset[dataset.process == 1]

        # For every unique MRN in the dataset as string
        for mrn in dataset.MRN.unique():
            # For every phonetic_ID in the subset
            for phonetic_id in dataset[dataset.MRN == mrn].Phonetic_ID.unique():

                print("Processing MRN {} Phonetic_ID {}".format(mrn, phonetic_id))
                
                patch_groups = {}

                # Get Patch_IDs for this phonetic_id
                patch_ids = dataset[(dataset.MRN == mrn) & (dataset.Phonetic_ID == phonetic_id)].Patch_ID.unique()
                
                # Group them into a dictionary by patch_id_stub where the stub is the patch ID without the trailing "-" and the number
                for patch_id in patch_ids:
                    # Skip patch_ids that are NaN
                    if(pd.isnull(patch_id)):
                        continue
                    # Remove the part after the last "-" if there are multiple "-" in the patch_id
                    patch_id_stub = patch_id.rsplit("-",1)[0]
                    if(patch_id_stub not in patch_groups):
                        patch_groups[patch_id_stub] = []
                    patch_groups[patch_id_stub].append(patch_id)
                
               

                # For every patch_id
                # Check the segmentations subfolder and see if the patch id stub is in there
                for patch_id_stub in patch_groups.keys():
                    # Get json files in the segmentation subfolder that start with the phonetic_id
                    json_files = glob.glob(os.path.join(DATASET_SUBFOLDER, "segmentation", phonetic_id + "*.json"))

                    # Get the json file that has the patch_id_stub in it
                    json_file = [json_file for json_file in json_files if patch_id_stub in open(json_file).read()]

                    # If no json file was found, skip this patch_id_stub
                    if(len(json_file) == 0):
                        print("No json file found for patch_id_stub {}".format(patch_id_stub))
                    else:
                        json_file = json_file[0]

                    #process_parameters.append([json_file, patch_id_stub, patch_groups, mrn, phonetic_id, DATASET_SUBFOLDER, dataset, df_process_log])

                    # Load the json file
                    with open(json_file) as f:
                        json_data = json.load(f)

                        # Get the SudyInstanceUID from the json file
                        StudyInstanceUID = json_data["StudyInstanceUID"]

                        # Parse the patch stub to get workflow ID (its part of the patch_stub before the first "-")
                        workflowId = patch_id_stub.split("-")[1]

                        # In the workflows section of the JSON file, find the correct workflow ID and gets its corresponding SeriesInstanceUID
                        SeriesInstanceUID = [workflow["SeriesInstanceUID"] for workflow in json_data["workflows"] if workflow["id"] == workflowId][0]

                        DICOM_Data_Path = os.path.join(DATASET_SUBFOLDER, "data", phonetic_id+"__"+StudyInstanceUID, StudyInstanceUID, SeriesInstanceUID)


                        # Check if this path exists, if not throw an error
                        if(not os.path.exists(DICOM_Data_Path)):
                            print ("DICOM data path {} does not exist".format(DICOM_Data_Path))
                            continue
                            #raise Exception("DICOM data path {} does not exist".format(DICOM_Data_Path))

                        # Load the scan annotaitons
                        annotations = ArterysAnnotations(json_file)

                        for patch_id in patch_groups[patch_id_stub]:
                            patch_features = annotations.getAnnotationsForPatch(patch_id)

                            if(patch_features is None):
                                print("No annotations found for patch {}".format(patch_id))
                                continue
                            
                            current_testcase = dataset[(dataset.MRN == mrn) & (dataset.Phonetic_ID == phonetic_id) & (dataset.Patch_ID == patch_id)]

                            table_row = {
                                'MRN': mrn,
                                'Phonetic_ID': phonetic_id,
                                'Patch_ID': patch_id,
                                'Slice_Thickness': current_testcase['thickness'].values[0],
                                'Accession_Number': current_testcase['Accession_Number'].values[0],
                                'label': current_testcase['label'].values[0],
                                'process': current_testcase['process'].values[0],
                                'volume': current_testcase['volume'].values[0],
                                'note': current_testcase['note'].values[0],
                                'thickness': current_testcase['thickness'].values[0],
                                'Detected_Arterys': current_testcase['Detected_Arterys'].values[0],	
                                'Series': current_testcase['Series'].values[0],
                                'Image': current_testcase['Image'].values[0],	
                                'Location': current_testcase['Location'].values[0],	
                                'Size': current_testcase['Size'].values[0],
                                'X': current_testcase['X'].values[0],                                'Y': current_testcase['Y'].values[0],
                                'Z': current_testcase['Z'].values[0],
                                'avg_pixel_value': patch_features['avg_pixel_value'],
                                'binary_compression': patch_features['binary_compression'],
                                'centroid_x': patch_features['centroid'][0][0],
                                'centroid_y': patch_features['centroid'][0][1],
                                'centroid_z': patch_features['centroid'][0][2],
                                'count': patch_features['count'],
                                'depth': patch_features['depth'],
                                'edited': patch_features['edited'],
                                'height': patch_features['height'],
                                'id': patch_features['id'],
                                'is_visible': patch_features['is_visible'],
                                'mask_code': patch_features['mask_code'],
                                'multi_component': patch_features['multi_component'],
                                'position_x': patch_features['position'][0][0],
                                'position_y': patch_features['position'][0][1],
                                'position_z': patch_features['position'][0][2],
                                'segmentation_type': patch_features['segmentation_type'],
                                'timepoint': patch_features['timepoint'],
                                'volume': patch_features['volume'],
                                'width': patch_features['width'],
                                'world_centroid_x': patch_features['world_centroid'][0][0],
                                'world_centroid_y': patch_features['world_centroid'][0][1],
                                'world_centroid_z': patch_features['world_centroid'][0][2],
                                'binary': patch_features['binary'],
                                'lld.major.distance': patch_features['lld.major.distance'],
                                'lld.major.p1_x': patch_features['lld.major.p1'][0][0],
                                'lld.major.p1_y': patch_features['lld.major.p1'][0][1],
                                'lld.major.p1_z': patch_features['lld.major.p1'][0][2],
                                'lld.major.p2_x': patch_features['lld.major.p2'][0][0],
                                'lld.major.p2_y': patch_features['lld.major.p2'][0][1],
                                'lld.major.p2_z': patch_features['lld.major.p2'][0][2],
                                'lld.minor.distance': patch_features['lld.minor.distance'],
                                'lld.minor.p1_x': patch_features['lld.minor.p1'][0][0],
                                'lld.minor.p1_y': patch_features['lld.minor.p1'][0][1],
                                'lld.minor.p1_z': patch_features['lld.minor.p1'][0][2],
                                'lld.minor.p2_x': patch_features['lld.minor.p2'][0][0],
                                'lld.minor.p2_y': patch_features['lld.minor.p2'][0][1],
                                'lld.minor.p2_z': patch_features['lld.minor.p2'][0][2]
                            }

                            # Add the patch_features to the dataframe
                            df_patch_features = pd.concat([df_patch_features,pd.DataFrame(table_row)], ignore_index=True)

    df_patch_features.to_csv("patch_features_new.csv", index=False)






    # # create a default thread pool
    # with mp.Pool(4) as pool:
    #     first_shard_id = 0

    #     for _ in tqdm.tqdm(pool.imap_unordered(processPatches, process_parameters), total=len(process_parameters)):
    #         pass
    #         #first_shard_id = tb.SaveBlocksToTFRecords(DATASET_FOLD,trainingBlocks,first_shard_id)
    #         #print("Finished processing scans with nodules.")
                    

                          
             

           