
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
    SaveImaged
)

from monai.data import  decollate_batch

import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.inferers import sliding_window_inference

from monai.losses import DiceCELoss, FocalLoss

from monai.metrics import DiceMetric, ROCAUCMetric,LossMetric

from monai.transforms import AsDiscrete

from tqdm import tqdm

from nodule_transforms.LoadArterysMaskd import LoadArterysMaskd
from nodule_transforms.AddCustomerMetadatad import AddCustomerMetadatad
from nodule_transforms.CropToNoduled import CropToNoduled
from nodule_transforms.EncodeClassIntoMask import EncodeClassIntoMask

import os
import torch
import numpy as np

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
PRE_PROCESSED_SUBFOLDER = '/home/vlermakov/Development/Nodules/Nodule-Detection-LUNA2016/processed_data_full_range_075mm/'
#PRE_PROCESSED_SUBFOLDER = './processed_data_unetr_range/'
#PRE_PROCESSED_SUBFOLDER = './processed_data_full_range_1mm/'


datalist = load_cancer_nodules_datalist(CANCER_NODULES_DATASET_DIR, PRE_PROCESSED_SUBFOLDER)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


train_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        LoadImaged(keys=["mask"], ensure_channel_first=True, image_only=False),
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "mask"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
        LoadImaged(keys=["mask"], ensure_channel_first=True, image_only=False),
    ]
)


train_data_list, val_data_list = balance_and_split_data(processed_scans,0.8)

train_data_list = train_data_list
val_data_list = val_data_list

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


# as explained in the "Setup transforms" section above, we want cached training images to not have metadata, and validations to have metadata
# the EnsureTyped transforms allow us to make this distinction
# on the other hand, set_track_meta is a global API; doing so here makes sure subsequent transforms (i.e., random transforms for training)
# will be carried out as Tensors, not MetaTensors
#set_track_meta(False)

case_num = 1
#img_name = os.path.split(train_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"].cpu()
mask = val_ds[case_num]["mask"].cpu()
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom model that will take SwinUNETR and pass its hidden layer to a fully connected layer
class SwinUNETRWithFC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(768, 1)

        # Average pooling to reduce the size of the image
        self.maxpool = torch.nn.AvgPool3d(kernel_size=3, stride=1)

    def forward(self, x):
        logit_map, hidden_layers = self.model(x)
        x = self.maxpool(hidden_layers[4]).view(-1, 768)
        return logit_map, self.fc(x)

# swin_model = SwinUNETR(
#     img_size=(96, 96, 96),
#     in_channels=1,
#     out_channels=2,
#     feature_size=48,
#     use_checkpoint=True,
# )


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

#model.load_state_dict(torch.load(os.path.join(root_dir, "saved_best_unet_classification.pth")))

# case_num = 4
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
#     img = val_ds[case_num]["image"]
#     label = val_ds[case_num]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
#     plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title("image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title("label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
#     plt.subplot(1, 3, 3)
#     plt.title("output")
#     plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
#     plt.show()

# exit(0)


torch.backends.cudnn.benchmark = True
loss_function_mask = DiceCELoss(to_onehot_y=True, softmax=True)
loss_function_label =  FocalLoss(gamma=2.0) #torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

def validation(epoch_iterator_val):

    losses = []
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_masks, val_label = (batch["image"].cuda(), batch["mask"].cuda(), batch["label"].cuda())

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
    loss = loss_function_mask(logit_map, y_mask)
    loss += 2*loss_function_label(logit_label.float(), y_label.unsqueeze(1).float())

    return loss

def train(global_step, train_loader, auc_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    loss_list = []
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y_mask,y_label = (batch["image"].cuda(), batch["mask"].cuda(), batch["label"].cuda())
        #with torch.cuda.amp.autocast():
        logit_map, logit_label = model(x)

        loss = compute_combined_loss(logit_map, logit_label, y_mask, y_label)
        loss_list.append(loss.item())
        
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={np.mean(loss_list):2.5f})")
    
    
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    dice_val,auc_val,mean_val_loss = validation(epoch_iterator_val)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    metric_values.append((dice_val,auc_val ))
    if auc_val > auc_val_best:
        auc_val_best = auc_val
        global_step_best = global_step
        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
        print(
            "Model Was Saved ! Current Best Avg. AUC: {} Current Avg. Dice: {} AUC: {} MeanLoss: {}".format(auc_val_best, dice_val, auc_val,mean_val_loss)
        )
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

case_num = 4
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    #img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    mask = val_ds[case_num]["mask"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_masks = torch.unsqueeze(mask, 1).cuda()
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
plt.imshow(torch.argmax(val_output_mask, dim=1).detach().cpu()[0, :, :, mask.shape[-1]//2])
plt.show()
