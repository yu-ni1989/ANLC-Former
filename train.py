import sys
import os

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import datetime
import argparse
import copy
import numpy as np
import pandas as pd
from dataloaders.StreamingDatasets import StreamingGeospatialDataset
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import utils_ANLG
import biformer_rdm

NUM_WORKERS = 4
NUM_CHIPS_PER_TILE = 100
CHIP_SIZE = 512

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--input_fn', type=str,
                    default=r'H:\code\dfc2021-msd-baseline-master\data\six\training_set_naip_nlcd_1v.csv',
                    help='输入cvs文件路径，包括三个文件夹。The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--input_val', type=str,
                    default=r'H:\code\dfc2021-msd-baseline-master\data\six\val_inference_1.csv')
parser.add_argument('--output_dir', type=str,
                    default=r'H:/Code/dfc2021-msd-baseline-master/results/biformer_rdm_r-ml-w/',
                    help='The path to a directory to store model checkpoints.')
parser.add_argument('--overwrite', action="store_true",
                    help='Flag for overwriting `output_dir` if that directory already exists.')
parser.add_argument('--save_most_recent', action="store_true", default='save_most_recent',
                    help='Flag for saving the most recent version of the model during training.')


## Training arguments
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size to use for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')  # 50
parser.add_argument('--seed', type=int, default=0, help='Random seed to pass to numpy and torch')
args = parser.parse_args()

def image_transforms(img, group):
    if group == 0:
        img = (img - utils_ANLG.NAIP_2013_MEANS) / utils_ANLG.NAIP_2013_STDS
    elif group == 1:
        img = (img - utils_ANLG.NAIP_2017_MEANS) / utils_ANLG.NAIP_2017_STDS
    else:
        raise ValueError("group not recognized")
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img


def label_transforms(labels):
    labels = utils_ANLG.NLCD_CLASS_TO_IDX_MAP[labels]
    labels = utils_ANLG.NLCD_IDX_TO_REDUCED_LC_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)
def val_nodata_check(img, labels):
    return np.any(labels > 3) or np.any(np.sum(img == 0, axis=2) == 4)

def main():
    print("Starting DFC2021 baseline training script at %s" % (str(datetime.datetime.now())))

    assert os.path.exists(args.input_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print(
                "WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (args.output_dir))
        else:
            print(
                "The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (
                    args.output_dir))
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print("WARNING! Torch is reporting that CUDA isn't available, exiting...")
        return

    input_dataframe = pd.read_csv(args.input_fn)  # 读取csv文件
    image_fns = input_dataframe["image_fn"].values  # 路径和文件连接
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    input_valdata = pd.read_csv(args.input_val)
    image_val = input_valdata["image_fn"].values
    label_val = input_valdata["val"].values
    groups_val = input_valdata["group"].values


    dataset = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=groups, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=False, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check
    )

    dataset_val = StreamingGeospatialDataset(
        imagery_fns=image_val, label_fns=label_val, groups=groups_val, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=False, verbose=False,
        image_transform=image_transforms, label_transform=None, nodata_check=val_nodata_check
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    ## added by yubin
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=2,
        num_workers=0,
        pin_memory=True,
    )

    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / args.batch_size)
    print("We will be training with %d batches per epoch" % (num_training_batches_per_epoch))
    strPathname = 'Biformer_rdm' + '_B' + str(args.batch_size) + '_'

    model = biformer_rdm.biformer_tiny()
    model = model.to(device)  # 载入模型
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)  # 优化器 lr=0.01
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    print("Model has %d parameters" % (utils_ANLG.count_parameters(model)))
    print("初始化的学习率：", optimizer.defaults['lr'])


    training_task_losses = []
    num_times_lr_dropped = 0
    model_checkpoints = []
    temp_model_fn = os.path.join(args.output_dir, "most_recent_model.pt")
    fa_i = open(args.output_dir + strPathname + "m_accuracy_values_b.txt", 'w')

    for epoch in range(args.num_epochs):
        lr = utils_ANLG.get_lr(optimizer)
        print("第%d个epoch的学习率：%f" % (epoch, lr))
        training_losses = utils_ANLG.fit(
            model,
            device,
            dataloader,
            num_training_batches_per_epoch,
            optimizer,
            criterion,
            epoch,
            dataloader_val,
            fa_i
        )
        # scheduler.step(training_losses[0])
        scheduler.step()

        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        if args.save_most_recent:
            torch.save(model.state_dict(), temp_model_fn)

        ###测试
        utils_ANLG.validation(
            model,
            device,
            dataloader_val,
            4,
            epoch,
            fa_i
        )

        if utils_ANLG.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")

        training_task_losses.append(training_losses[0])

        if num_times_lr_dropped == 4:
            break


    save_obj = {
        'args': args,
        'training_task_losses': training_task_losses,
        "checkpoints": model_checkpoints
    }

    save_obj_fn = "results.pt"
    with open(os.path.join(args.output_dir, save_obj_fn), 'wb') as f:
        torch.save(save_obj, f)


if __name__ == "__main__":
    main()
