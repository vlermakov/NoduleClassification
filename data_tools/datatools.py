import random
import glob
import os
import json

import pandas as pd

import torch


def upsample_minority_class(data):
    # Get the number of positive and negative samples
    num_positive_samples = len([item for item in data if item['label'] == 1])
    num_negative_samples = len([item for item in data if item['label'] == 0])

    # Get the ratio of positive to negative samples
    ratio = num_positive_samples / num_negative_samples

    # If the ratio is 1, no need to upsample
    if ratio == 1:
        return data
    
    # If the ratio is less than 1, upsample the positive samples
    if ratio < 1:
        positive_samples = [item for item in data if item['label'] == 1]
        negative_samples = [item for item in data if item['label'] == 0]
        positive_samples_upsampled = random.choices(positive_samples, k=num_negative_samples)
        return positive_samples_upsampled + negative_samples


def balance_and_split_data(data, train_set_proportion = 0.5):
    # Upsample the minority class
    data = upsample_minority_class(data)

    # Split the data into test and validation sets
    test_set,validation_set = split_data(data,train_set_proportion)

    # Remove items that have duplicate patch ids in the validation set
    validation_set = pd.DataFrame(validation_set)
    validation_set = validation_set.drop_duplicates(subset=['patch_id'], keep='first')
    validation_set = validation_set.to_dict('records') 

    return test_set, validation_set




def split_data(data, train_set_proportion=0.5):
    # Group data by MRN
    mrn_groups = {}
    for item in data:
        if item['mrn'] not in mrn_groups:
            mrn_groups[item['mrn']] = []
        mrn_groups[item['mrn']].append(item)
    
    # Convert groups to list and shuffle
    grouped_data = list(mrn_groups.values())
    random.shuffle(grouped_data)

    test_set = []
    validation_set = []

    label_counts_test = {}
    label_counts_validation = {}

    for group in grouped_data:
        label_for_group = group[0]['label']  # assuming each group has the same label

        # Initialize label counts if not already present
        label_counts_test.setdefault(label_for_group, 0)
        label_counts_validation.setdefault(label_for_group, 0)

        # Decide where to place group based on current label balance taking into account test_set_size
        if label_counts_test[label_for_group] / (label_counts_test[label_for_group] + label_counts_validation[label_for_group]+0.0001) < train_set_proportion:
            test_set.extend(group)
            label_counts_test[label_for_group] += len(group)
        else:
            validation_set.extend(group)
            label_counts_validation[label_for_group] += len(group)


    return test_set, validation_set

def folder_exists_with_partial_name(search_path, partial_name):
    """
    Checks if a folder starting with the given partial_name exists within the search_path.
    
    :param search_path: The path to search within.
    :param partial_name: The starting name of the folder to search for.
    :return: True if a matching folder is found, else False.
    """

    # Check if search_path exists
    if not os.path.exists(search_path):
        return False
    
    for folder_name in os.listdir(search_path):
        if folder_name.startswith(partial_name) and os.path.isdir(os.path.join(search_path, folder_name)):
            return True
    return False

def load_LUNA16_datalist(LUNA16_DATASET_DIR, PROCESSED_SUBFOLDER = None):
    LUNA16_datalist = []

    # Get the list of all the .mhd files in all the subfolders
    mhd_files = glob.glob(os.path.join(LUNA16_DATASET_DIR, "*subset*", "*.mhd"), recursive=True)

    # For every .mhd file
    for mhd_file in mhd_files:
        # Take the filename without the extension as the series id. Filename contains multiple "." symbols, so make sure we only strip off the extension after last "."
        series_id = os.path.basename(mhd_file).rsplit(".", 1)[0]

        # Split the path to get the subset id (subset0, subset1, etc) from the path
        subset_id = os.path.basename(os.path.dirname(mhd_file))


        # Check if there is a folder that starts with folder_partial_name exists
        if PROCESSED_SUBFOLDER is None or (not folder_exists_with_partial_name(PROCESSED_SUBFOLDER, subset_id + "__" + series_id)):
            scan_path = mhd_file
            mask_path = series_id
            pre_processed_status = False
            label = 0
            LUNA16_datalist.append({
                'mask': mask_path,
                'image':  scan_path,
                'pre_processed': pre_processed_status,
                'series_id': series_id,
                'subset_id': subset_id,
                'label': label
            })
        else:
            # Find folder in the PROCESSED_SUBFOLDER that starts with the subset_id and series_id
            processed_path_to_scan = [folder_name for folder_name in os.listdir(PROCESSED_SUBFOLDER) if folder_name.startswith(subset_id + "__" + series_id)][0]

            # Find the scan and mask files in the processed folder
            scan_path = os.path.join(PROCESSED_SUBFOLDER, processed_path_to_scan, processed_path_to_scan+"_image_0.nii.gz")
            mask_path = os.path.join(PROCESSED_SUBFOLDER, processed_path_to_scan, processed_path_to_scan+"_mask_0.nii.gz")

            #series_id_numeric = series_id.split(".")[-1]
            # scan_path = os.path.join(PROCESSED_SUBFOLDER, os.path.basename(mhd_file), os.path.basename(mhd_file)+"_image_0.nii.gz")
            # mask_path = os.path.join(PROCESSED_SUBFOLDER, os.path.basename(mhd_file), os.path.basename(mhd_file)+"_mask_0.nii.gz")
            pre_processed_status = True

            # Get the label for this series_id and subset_id. To do this find the folder that starts with the subset_id and series_id and get the label from remaining part of the folder name
            # that will be in format "_label_1", "_label_2" "_label_0"
            label = [folder_name for folder_name in os.listdir(PROCESSED_SUBFOLDER) if folder_name.startswith(subset_id + "__" + series_id)][0].rsplit("_", 1)[1]

            if(label != "A"):
                label_float = float(label) 
                if(label_float <= 2.5 or label_float >= 3.5):
                    label = 1 if label_float >= 3.5 else 0
                    LUNA16_datalist.append({
                        'mask': mask_path,
                        'image':  scan_path,
                        'pre_processed': pre_processed_status,
                        'series_id': series_id,
                        'subset_id': subset_id,
                        'label': torch.tensor(label).type(torch.float32)  # Convert labels to 0 and 1
                        })

    return LUNA16_datalist

def load_cancer_nodules_datalist(CANCER_NODULES_DATASET_DIR, PROCESSED_SUBFOLDER = None):

    # Read in the dataset definition
    dataset = pd.read_csv(os.path.join(CANCER_NODULES_DATASET_DIR, "dataset.csv"), dtype={'MRN': str})

    return load_cancer_nodules_datalist_from_dataframe(dataset, CANCER_NODULES_DATASET_DIR, PROCESSED_SUBFOLDER)

def load_cancer_nodules_datalist_from_dataframe(dataset, CANCER_NODULES_DATASET_DIR, PROCESSED_SUBFOLDER = None):
    cancer_nodules_datalist = []

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
                # Check if patch id is not a string
                if(not isinstance(patch_id, str)):
                    print("Patch ID {} is not a string".format(patch_id))
                    continue
            
                # Get json files in the segmentation subfolder that start with the phonetic_id
                json_files = glob.glob(os.path.join(CANCER_NODULES_DATASET_DIR, "segmentation", phonetic_id + "*.json"))

                patch_id_stub = patch_id.rsplit("-",1)[0]

                # Get the json file that has the patch_id_stub in it
                json_file = [json_file for json_file in json_files if patch_id_stub in open(json_file).read()]

                # If no json file was found, skip this patch_id_stub
                if(len(json_file) == 0):
                    print("No json file found for patch_id_stub {}".format(patch_id_stub))
                    continue
                else:
                    json_file = json_file[0]

                with open(json_file) as f:
                    json_data = json.load(f)

                # Get the SudyInstanceUID from the json file
                StudyInstanceUID = json_data["StudyInstanceUID"]

                # Parse the patch stub to get workflow ID (its part of the patch_stub before the first "-")
                workflowId = patch_id_stub.split("-")[1]

                # In the workflows section of the JSON file, find the correct workflow ID and gets its corresponding SeriesInstanceUID
                SeriesInstanceUID = [workflow["SeriesInstanceUID"] for workflow in json_data["workflows"] if workflow["id"] == workflowId][0]

                DICOM_Data_Path = os.path.join(CANCER_NODULES_DATASET_DIR, "data", phonetic_id+"__"+StudyInstanceUID, StudyInstanceUID, SeriesInstanceUID)

                # Check if the patch was already processed
                if PROCESSED_SUBFOLDER is None or (not os.path.exists(os.path.join(PROCESSED_SUBFOLDER, phonetic_id+"__"+patch_id))):
                    scan_path = DICOM_Data_Path
                    mask_path = json_file
                    pre_processed_status = False
                else:
                    #series_id_numeric = StudyInstanceUID.split(".")[-1]
                    scan_path = os.path.join(PROCESSED_SUBFOLDER, phonetic_id+"__"+patch_id,phonetic_id+"__"+patch_id+"_image_0.nii.gz")
                    mask_path = os.path.join(PROCESSED_SUBFOLDER, phonetic_id+"__"+patch_id,phonetic_id+"__"+patch_id+"_mask_0.nii.gz")
                    pre_processed_status = True

                cancer_nodules_datalist.append({
                    'mask': mask_path,
                    'image':  scan_path,
                    'label': dataset[dataset.Patch_ID == patch_id].label.values[0],
                    'phonetic_id': phonetic_id,
                    'patch_id': patch_id,
                    'mrn': mrn,
                    'pre_processed': pre_processed_status})
                
    return cancer_nodules_datalist
