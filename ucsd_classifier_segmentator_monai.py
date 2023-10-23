import wandb


WANDB_PROJECT_NAME = "NoduleClassification"

#87e3b4f079c5a73e70e9ebd3bda9dc48ee553d9f
wandb.login(key='87e3b4f079c5a73e70e9ebd3bda9dc48ee553d9f')


import tempfile

from monai.visualize.utils import matshow3d, blend_images
from monai.data import CacheDataset, ThreadDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    BorderPad,
    SaveImaged,
    RandRotated,
    Rotated
)

from monai.data import  decollate_batch

import matplotlib.pyplot as plt
from basic_unet import BasicUNet
from monai.inferers import sliding_window_inference

from monai.losses import DiceCELoss, FocalLoss

from monai.metrics import DiceMetric, ROCAUCMetric,LossMetric

from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.transforms import AsDiscrete

from tqdm import tqdm

from nodule_transforms.LoadArterysMaskd import LoadArterysMaskd
from nodule_transforms.AddCustomerMetadatad import AddCustomerMetadatad
from nodule_transforms.CropToNoduled import CropToNoduled
from nodule_transforms.EncodeClassIntoMask import EncodeClassIntoMask

import os
import torch
import numpy as np
import pandas as pd


from data_tools.datatools import load_cancer_nodules_datalist, split_data, balance_and_split_data


# Save random seed
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

directory = './monai_data'
if not os.path.exists(directory):
    os.makedirs(directory)
    
root_dir = tempfile.mkdtemp() if directory is None else directory
cache_dir = './training_cache'

CANCER_NODULES_DATASET_DIR = '/media/vlermakov/data/UCSDNodules/Metastatic/'
#PRE_PROCESSED_DIRECTORY = '/home/vlermakov/Development/Nodules/Nodule-Detection-LUNA2016/processed_data_full_range/'
PRE_PROCESSED_SUBFOLDER = '/home/vlermakov/Development/Nodules/Nodule-Detection-LUNA2016/processed_data_full_range_075mm_fixed/'
#PRE_PROCESSED_SUBFOLDER = './processed_data_unetr_range/'
#PRE_PROCESSED_SUBFOLDER = './processed_data_full_range_1mm/'


dataset = pd.read_csv(os.path.join(CANCER_NODULES_DATASET_DIR, "dataset.csv"), dtype={'MRN': str})

# Ignore class = 1
#dataset = dataset[dataset.label != 1]

# All class 2 set to class 1
dataset.loc[dataset.label == 2, 'label'] = 1

datalist = load_cancer_nodules_datalist(CANCER_NODULES_DATASET_DIR, PRE_PROCESSED_SUBFOLDER)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = "cuda" if torch.cuda.is_available() else "cpu"


def file_name_formatter(metadict: dict, saver) -> dict:
   return {"subject": metadict['phonetic_id']+"__"+metadict['patch_id'], "idx": 0}

processed_scans = []
unprocessed_scans = []

n_processed = 0
for p in datalist:
    if p['pre_processed']:
        n_processed += 1 
        processed_scans.append(p)
    else:
        unprocessed_scans.append(p)

if(False and len(unprocessed_scans) > 0):
    pre_process_transform = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
            LoadArterysMaskd(keys=["mask"], ensure_channel_first=True),
            #AddCustomerMetadatad(keys=["image","mask"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            #Orientationd(keys=["image","mask"], axcodes="RAS"),
            Spacingd(
                keys=["image","mask"],
                pixdim=(0.75, 0.75, 0.75),
                mode=("bilinear","nearest"),
            ),
            CropToNoduled(keys=["image","mask"]),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            # SaveImaged(keys=["image"], output_name_formatter=file_name_formatter ,output_dir="./output_training", output_postfix="image", output_ext=".nii.gz", resample=False),
            # SaveImaged(keys=["mask"], output_name_formatter=file_name_formatter, output_dir="./output_training", output_postfix="mask", output_ext=".nii.gz", resample=False),
        ]
    )

    train_ds = CacheDataset(
        data=unprocessed_scans,
        transform=pre_process_transform,
        cache_num=len(unprocessed_scans),
        cache_rate=1.0,
        num_workers=8,
    )


train_data_list, val_data_list = balance_and_split_data(processed_scans,0.8)
train_df = pd.DataFrame(train_data_list)
val_df = pd.DataFrame(val_data_list)

# Count disticnt MRNs in train and validation data
nPatientsTrain = len(pd.unique(train_df['mrn']))
nPatientsVal = len(pd.unique(val_df['mrn']))

nClassOnePatientsTrain = len(pd.unique(train_df[train_df['label']==1]['mrn']))
nClassOnePatientsVal = len(pd.unique(val_df[val_df['label']==1]['mrn']))

print(f"Number of positive patients in train data: {nClassOnePatientsTrain} /  {nPatientsTrain}")
print(f"Number of positive patients in validation data: {nClassOnePatientsVal} / {nPatientsVal}")

nTrain = len(train_df)
nVal = len(val_df)

nClassOneTrain = len(train_df[train_df['label']==1])
nClassOneVal = len(val_df[val_df['label']==1])

print(f"Number of positive samples in train data: {nClassOneTrain} /  {nTrain}")
print(f"Number of positive samples in validation data: {nClassOneVal} / {nVal}")


run_parameters = {
    "dataset_folder": CANCER_NODULES_DATASET_DIR,
    "pre_processed_folder": PRE_PROCESSED_SUBFOLDER,
    "dataset_artifact": "nodules_and_masks_dataset",

    "classification_factor": 30,
    "learning_rate": 0.0001,
    "patience_for_lr_scheduler": 5,
    "factor_for_lr_scheduler": 0.5,
    "loss_function": "FocalLoss",
}

run = wandb.init(
        # Set the project where this run will be logged
        project=WANDB_PROJECT_NAME,
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"SquamousPlusAdenoVsMetastatic_v4",
        job_type="training",
        # Track hyperparameters and run metadata
        config=run_parameters
        )


dataset_table = wandb.Table(dataframe=pd.DataFrame(datalist))

dataset_artifact = wandb.Artifact(
    'datalist_table',
    type="dataset"
    )
dataset_artifact.add(dataset_table, "datalist_table")

# Log the raw csv file within an artifact to preserve our data
dataset_artifact.add_file(os.path.join(run.config.dataset_folder, 'dataset.csv'))

# Log the table to visualize with a run...
run.log({"dataset_table": dataset_table})
run.log_artifact(dataset_artifact)

train_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        LoadImaged(keys=["mask"], ensure_channel_first=True, image_only=False),
        # RandFlipd(
        #     keys=["image", "mask"],
        #     spatial_axis=[0],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "mask"],
        #     spatial_axis=[1],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "mask"],
        #     spatial_axis=[2],
        #     prob=0.10,
        # ),
        # RandRotate90d(
        #     keys=["image", "mask"],
        #     prob=0.10,
        #     max_k=3,
        # ),
        Rotated(keys=["image", "mask"], angle=0.1, mode=["bilinear", "nearest"]),
        RandRotated(
                keys=["image", "mask"],
                prob=0.99,
                range_x=np.pi/6,
                range_y=np.pi/6,
                range_z=np.pi/6,
                mode=["bilinear", "nearest"]),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=1,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        LoadImaged(keys=["mask"], ensure_channel_first=True, image_only=False),
    ]
)

train_ds = CacheDataset(
    data=train_data_list,
    transform=train_transforms,
    cache_num=len(train_data_list),
    cache_rate=1.0,
    num_workers=8,
)

train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
val_ds = CacheDataset(data=val_data_list, transform=val_transforms, cache_num=len(val_data_list), cache_rate=1.0, num_workers=4)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)



train_data_list_table = wandb.Table(dataframe=pd.DataFrame(train_data_list))

train_data_artifact = wandb.Artifact(
    'train_data_list_table',
    type="dataset"
    )

train_data_artifact.add(train_data_list_table, "train_data_list_table")

# Log the table to visualize with a run...
run.log({"train_data_list_table": train_data_list_table})

# Log the table to visualize with a run...
run.log_artifact(train_data_artifact)


validation_data_list_table = wandb.Table(dataframe=pd.DataFrame(val_data_list))

validation_data_artifact = wandb.Artifact(
    'validation_data_list_table',
    type="dataset"
    )

validation_data_artifact.add(validation_data_list_table, "validation_data_list_table")

# Log the table to visualize with a run...
run.log({"validation_data_list_table": validation_data_list_table})
run.log_artifact(validation_data_artifact)


# as explained in the "Setup transforms" section above, we want cached training images to not have metadata, and validations to have metadata
# the EnsureTyped transforms allow us to make this distinction
# on the other hand, set_track_meta is a global API; doing so here makes sure subsequent transforms (i.e., random transforms for training)
# will be carried out as Tensors, not MetaTensors
#set_track_meta(False)

case_num = 0
testcase = train_ds[case_num]
img = testcase["image"].cpu()
mask = testcase["mask"].cpu()
img_shape = img.shape
mask_shape = mask.shape
print(f"image shape: {img_shape}, mask shape: {mask_shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, img_shape[-1]//2].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("mask")
plt.imshow(mask[0, :, :, img_shape[-1]//2].detach().cpu())
plt.show()



blended_image = blend_images(img, mask, alpha=0.5, cmap="hsv")
# Generate two plot windows with two figures
fig1 = plt.figure()

matshow3d(blended_image, channel_dim = 0, fig = fig1, frame_dim=-1)
plt.show()



unet_model = BasicUNet(
    #img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14
    #feature_size=48,
    #use_checkpoint=True,
).to(device)


# Define a custom model that will take SwinUNETR and pass its hidden layer to a fully connected layer
class BasicUNETRWithFC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(256, 1)

        # Average pooling to reduce the size of the image
        self.pooling = torch.nn.AvgPool3d(kernel_size=6, stride=1)

    def forward(self, x):
        logit_map, hidden_layers = self.model(x)
        x = self.pooling(hidden_layers).view(-1, 256)
        x = self.fc(x)
        classification_output = torch.nn.functional.sigmoid(x).to(dtype=torch.float32)

        

        return logit_map, classification_output

model = BasicUNETRWithFC(unet_model).to(device)



segmentation_factor = 1

torch.backends.cudnn.benchmark = True
loss_function_mask = DiceCELoss(to_onehot_y=True, softmax=True)
#loss_function_label =  torch.nn.BCELoss()
loss_function_label =  FocalLoss(gamma=2.0) #torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=run.config.learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=run.config.patience_for_lr_scheduler, factor=run.config.factor_for_lr_scheduler, verbose = True)

#scaler = torch.cuda.amp.GradScaler()

def validation(epoch_iterator_val):

    losses = []
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_masks, val_label = (batch["image"].to(device), batch["mask"].to(device), batch["label"].to(device))

            #with torch.cuda.amp.autocast():
            val_output_mask, val_output_label = model(val_inputs)

            val_loss = compute_combined_loss(val_output_mask, val_output_label, val_masks, val_label)

            losses.append(val_loss.item())

            val_masks_list = decollate_batch(val_masks)
            val_masks_convert = [post_label(val_mask_tensor) for val_mask_tensor in val_masks_list]
            
            val_output_mask_list = decollate_batch(val_output_mask)
            val_output_mask_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_output_mask_list]

            predicted_classes = torch.argmax(val_output_mask, dim=1).type(torch.int32).as_tensor().detach().cpu()
            mean_predicted_class = predicted_classes[predicted_classes != 0].float().mean()
            actual_class = torch.max(val_masks).detach().cpu()

            roc_auc_metric( y_pred=val_output_label.squeeze(-1), y=val_label)
            
            dice_metric(y_pred=val_output_mask_convert, y=val_masks_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        mean_auc_val = roc_auc_metric.aggregate().item()
        dice_metric.reset()
        roc_auc_metric.reset()
    return mean_dice_val, mean_auc_val, np.mean(losses)


def compute_combined_loss(logit_map, logit_label, y_mask, y_label):
    loss = segmentation_factor*loss_function_mask(logit_map, y_mask)
    loss += run.config.classification_factor*loss_function_label(logit_label.float(), y_label.unsqueeze(1).float())

    return loss

def train(global_step, train_loader, auc_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    loss_list = []
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y_mask,y_label = (batch["image"].to(device), batch["mask"].to(device), batch["label"].to(device))
        #with torch.cuda.amp.autocast():
        logit_map, logit_label = model(x)

        loss = compute_combined_loss(logit_map, logit_label, y_mask, y_label)
        loss_list.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        #scaler.scale(loss).backward()FIND was unable to find an engine to execute this computation
        
        optimizer.step()

        epoch_loss += loss.item()
        # scaler.unscale_(optimizer)
        # scaler.step(optimizer)
        # scaler.update()
        
        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={np.mean(loss_list):2.5f})")
    
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    dice_val,auc_val,mean_val_loss = validation(epoch_iterator_val)

    wandb.log({
        'loss': np.mean(loss_list),
        'val_loss': mean_val_loss,
        'val_dice': dice_val,
        'val_auc': auc_val
    })

    scheduler.step(mean_val_loss)


    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    metric_values.append((dice_val,auc_val ))
    if auc_val > auc_val_best:
        auc_val_best = auc_val
        global_step_best = global_step
        model_file = os.path.join(root_dir, "best_metric_model.pth")
        torch.save(model.state_dict(), model_file)
        print(
            "Model Was Saved ! Current Best Avg. AUC: {} Current Avg. Dice: {} AUC: {} MeanLoss: {}".format(auc_val_best, dice_val, auc_val,mean_val_loss)
        )


        artifact = wandb.Artifact(
          name='segment_and_detect',
          type='model'
        )

        artifact.add_file(local_path=model_file)

        run.log_artifact(artifact)

        patience = 0

    else:
        print(
            "Model Was Not Saved ! Current Best Avg. AUC: {} Current Avg. Dice: {} AUC: {} MeanLoss: {}".format(
                auc_val_best, dice_val, auc_val, mean_val_loss
            )
        )
    global_step += 1
    return global_step, auc_val_best, global_step_best


max_iterations = 30000
eval_num = 1000
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
loss_metric = LossMetric(loss_fn=compute_combined_loss)
roc_auc_metric = ROCAUCMetric()

global_step = 0
dice_val_best = 0.0
auc_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


while global_step < max_iterations:
    global_step, auc_val_best, global_step_best = train(global_step, train_loader, auc_val_best, global_step_best)
    # if(global_step > 10):
    #   print("Turning off mask optimization")
    #   segmentation_factor = 0

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

print(f"train completed, best_metric: {auc_val_best:.4f} " f"at iteration: {auc_val_best}")

plt.figure("train", (12, 6))
plt.subplot(1, 3, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = [m[0] for m in metric_values]
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 3, 3)
plt.title("Val Mean AUC")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = [m[1] for m in metric_values]
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()

display_ds = val_ds

already_seen = []
for case_num in range(0, len(display_ds), 1):
  #print(f"cnt: {case_num}")
  data_case = display_ds[case_num]
  if(int(data_case["label"]) != 1):
    continue

  if(data_case["phonetic_id"] in already_seen):
    continue
  else:
    already_seen.append(data_case["phonetic_id"])

  model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
  model.eval
  already_seen.append(data_case["phonetic_id"])

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    #img_name = os.path.split(data_case["image"].meta["filename_or_obj"])[1]
    # Add title
    img = data_case["image"]
    mask = data_case["mask"]
    label = data_case["label"]
    val_inputs = torch.unsqueeze(img, 1).to(device)
    val_masks = torch.unsqueeze(mask, 1).to(device)
    val_output_mask,val_output_label = model(val_inputs) #sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
plt.figure("check", (18, 6))
plt.subplot(1, 3, 1)
plt.title("image")
plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, img.shape[-1]//2], cmap="gray")
plt.subplot(1, 3, 2)
plt.title("mask")
plt.imshow(val_masks.cpu().numpy()[0, 0, :, :, mask.shape[-1]//2])
plt.subplot(1, 3, 3)
plt.title("output")

plt.suptitle(str(case_num) + ": " + val_ds[case_num]["phonetic_id"] + " True: " + str(label) + " Predicted: " + str(val_output_label.detach().cpu().float()))
plt.imshow(torch.argmax(val_output_mask, dim=1).detach().cpu()[0, :, :, mask.shape[-1]//2])
plt.show()
