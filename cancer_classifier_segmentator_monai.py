
import tempfile

from monai.visualize.utils import matshow3d, blend_images
from monai.data import CacheDataset, ThreadDataLoader, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    BorderPad,
    SaveImaged,
    Lambdad,
)

from monai.data import  decollate_batch

import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

from monai.losses import DiceCELoss

from monai.metrics import DiceMetric, ROCAUCMetric

from monai.transforms import AsDiscrete

from tqdm import tqdm

from nodule_transforms.EncodeClassIntoMask import EncodeClassIntoMask




import os
import torch
import numpy as np
import pandas as pd


from data_tools.datatools import load_cancer_nodules_datalist, split_data, balance_and_split_data


directory = './monai_data'
root_dir = tempfile.mkdtemp() if directory is None else directory
cache_dir = './training_cache'

CANCER_NODULES_DATASET_DIR = '/media/vlermakov/data/UCSDNodules/Metastatic/'
#PRE_PROCESSED_DIRECTORY = '/home/vlermakov/Development/Nodules/Nodule-Detection-LUNA2016/processed_data_full_range/'
PRE_PROCESSED_SUBFOLDER = './processed_data_full_range_075mm/'


datalist = load_cancer_nodules_datalist(CANCER_NODULES_DATASET_DIR, PRE_PROCESSED_SUBFOLDER)


# Load validation patch ids from the CSV file
validation_patch_ids = pd.read_csv(os.path.join("validation_patch_ids.csv"))

# Select lines from datalist that have patch_ids in the validation_patch_ids and are processed
val_data_list = [item for item in datalist if item['patch_id'] in validation_patch_ids.patch_id.values and item['pre_processed'] == 1]


val_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        LoadImaged(keys=["mask"], ensure_channel_first=True, image_only=False),
        EncodeClassIntoMask(keys=["mask"]),
    ]
)

val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_num=len(val_data_list), cache_rate=1.0, num_workers=4)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model.load_state_dict(torch.load(os.path.join(root_dir, "saved_best_metric_model.pth")))


# case_num = 4
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     #img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
#     img = val_ds[case_num]["image"]
#     mask = val_ds[case_num]["mask"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_masks = torch.unsqueeze(mask, 1).cuda()
#     val_output_mask = model(val_inputs) #sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
# plt.figure("check", (18, 6))
# plt.subplot(1, 3, 1)
# plt.title("image")
# plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, img.shape[-1]//2], cmap="gray")
# plt.subplot(1, 3, 2)
# plt.title("mask")
# plt.imshow(val_masks.cpu().numpy()[0, 0, :, :, mask.shape[-1]//2])
# plt.subplot(1, 3, 3)
# plt.title("output")
# plt.imshow(torch.argmax(val_output_mask, dim=1).detach().cpu()[0, :, :, mask.shape[-1]//2])
# plt.show()


post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
true_dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
roc_auc_metric = ROCAUCMetric()

true_labels = []
pred_labels = []

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_masks = (batch["image"].cuda(), batch["mask"].cuda())
            with torch.cuda.amp.autocast():
                val_output_mask = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_masks_list = decollate_batch(val_masks)
            val_masks_convert = [post_label(val_mask_tensor) for val_mask_tensor in val_masks_list]
            val_masks_merged = [torch.max(post_label(val_mask_tensor)[1:],dim=0).values for val_mask_tensor in val_masks_list]
            
            val_output_mask_list = decollate_batch(val_output_mask)
            val_output_mask_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_output_mask_list]
            val_output_mask_merged = [torch.max(post_pred(val_pred_tensor)[1:],dim=0).values for val_pred_tensor in val_output_mask_list]

            predicted_classes = torch.argmax(val_output_mask, dim=1).type(torch.int32).as_tensor().detach().cpu()
            mean_predicted_class = predicted_classes[predicted_classes != 0].float().mean()
            actual_class = torch.max(val_masks).detach().cpu()

            roc_auc_metric( y_pred=torch.tensor([mean_predicted_class-1.0]), y=torch.tensor([actual_class-1.0]))

            #print(f"   Predicted Class: {mean_predicted_class} Actual Class: {actual_class}")
            
            dice_metric(y_pred=val_output_mask_convert, y=val_masks_convert)

            true_dice_metric(y_pred=val_output_mask_merged, y=val_masks_merged)



            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        mean_true_dice_val = true_dice_metric.aggregate().item()
        mean_auc = roc_auc_metric.aggregate().item()
        dice_metric.reset()
        true_dice_metric.reset()
        roc_auc_metric.reset()
    return mean_dice_val, mean_true_dice_val, mean_auc

epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
dice_val, true_dice_val, mean_auc_val = validation(epoch_iterator_val)

# Compute auc based on true_labels and pred_labels

print("AUC: ", mean_auc_val)



print("Dice: ", dice_val)
print("True Dice: ", true_dice_val)