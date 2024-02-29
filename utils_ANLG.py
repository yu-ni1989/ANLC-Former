import sys
import time
import numpy as np
import ot

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import DPatSegMetrics as metrics

NAIP_2013_MEANS = np.array([117.00, 130.75, 122.50, 159.30])
NAIP_2013_STDS = np.array([38.16, 36.68, 24.30, 66.22])
NAIP_2017_MEANS = np.array([72.84, 86.83, 76.78, 130.82])
NAIP_2017_STDS = np.array([41.78, 34.66, 28.76, 58.95])
NLCD_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90,
                95]  # 16 classes + 1 nodata class ("0"). Note that "12" is "Perennial Ice/Snow" and is not present in Maryland.

NLCD_CLASS_COLORMAP = {  # Copied from the emebedded color table in the NLCD data files
    0: (0, 0, 0, 255),
    11: (70, 107, 159, 255),
    12: (209, 222, 248, 255),
    21: (222, 197, 197, 255),
    22: (217, 146, 130, 255),
    23: (235, 0, 0, 255),
    24: (171, 0, 0, 255),
    31: (179, 172, 159, 255),
    41: (104, 171, 95, 255),
    42: (28, 95, 44, 255),
    43: (181, 197, 143, 255),
    52: (204, 184, 121, 255),
    71: (223, 223, 194, 255),
    81: (220, 217, 57, 255),
    82: (171, 108, 40, 255),
    90: (184, 217, 235, 255),
    95: (108, 159, 184, 255)
}

LC4_CLASS_COLORMAP = {
    0: (0, 0, 255, 255),
    1: (0, 128, 0, 255),
    2: (128, 255, 128, 255),
    3: (128, 96, 96, 255),
    4: (0, 0, 0, 255)
}

NLCD_IDX_COLORMAP = {
    idx: NLCD_CLASS_COLORMAP[c]
    for idx, c in enumerate(NLCD_CLASSES)
}


def get_nlcd_class_to_idx_map():
    nlcd_label_to_idx_map = []
    idx = 0
    for i in range(NLCD_CLASSES[-1] + 1):
        if i in NLCD_CLASSES:
            nlcd_label_to_idx_map.append(idx)
            idx += 1
        else:
            nlcd_label_to_idx_map.append(0)
    nlcd_label_to_idx_map = np.array(nlcd_label_to_idx_map).astype(np.int64)
    return nlcd_label_to_idx_map


NLCD_CLASS_TO_IDX_MAP = get_nlcd_class_to_idx_map()  # I do this computation on import for illustration (this could instead be a length 96 vector that is hardcoded here)

NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,  # 0 No data 0
    0,  # 1 Open Water
    4,  # 2 Ice/Snow
    2,  # 3 Developed Open Space
    3,  # 4 Developed Low Intensity
    3,  # 5 Developed Medium Intensity
    3,  # 6 Developed High Intensity
    3,  # 7 Barren Land
    1,  # 8 Deciduous Forest
    1,  # 9 Evergreen Forest
    1,  # 10 Mixed Forest
    1,  # 11 Shrub/Scrub
    2,  # 12 Grassland/Herbaceous
    2,  # 13 Pasture/Hay
    2,  # 14 Cultivated Crops
    1,  # 15 Woody Wetlands
    1,  # 16 Emergent Herbaceious Wetlands
])

NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR = np.array([
    [0, 0, 0, 0, 1],  # 0 No data 0
    [1, 0, 0, 0, 0],  # 1 Open Water
    [0, 0, 0, 0, 1],  # 2 Ice/Snow
    [0, 0, 0, 0, 0],  # 3 Developed Open Space
    [0, 0, 0, 0, 0],  # 4 Developed Low Intensity
    [0, 0, 0, 1, 0],  # 5 Developed Medium Intensity
    [0, 0, 0, 1, 0],  # 6 Developed High Intensity
    [0, 0, 0, 0, 0],  # 7 Barren Land
    [0, 1, 0, 0, 0],  # 8 Deciduous Forest
    [0, 1, 0, 0, 0],  # 9 Evergreen Forest
    [0, 1, 0, 0, 0],  # 10 Mixed Forest
    [0, 1, 0, 0, 0],  # 11 Shrub/Scrub
    [0, 0, 1, 0, 0],  # 12 Grassland/Herbaceous
    [0, 0, 1, 0, 0],  # 13 Pasture/Hay
    [0, 0, 1, 0, 0],  # 14 Cultivated Crops
    [0, 1, 0, 0, 0],  # 15 Woody Wetlands
    [0, 1, 0, 0, 0],  # 16 Emergent Herbaceious Wetlands
])

def ot_mask(output, label):
    B, C, W, H = output.shape
    mask = torch.zeros(B, 1, W, H)
    # 将label转换为One-Hot编码
    label = label.to(torch.int64).cpu()
    one_hot = F.one_hot(label, num_classes=4)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    # 使用双线性插值下采样到(8, 4, 32, 32)
    downsampled_output = F.interpolate(output, size=(32, 32), mode='bilinear', align_corners=False)
    downsampled_label = F.interpolate(one_hot, size=(32, 32), mode='nearest')

    # 循环计算掩膜
    for i in range(0, downsampled_output.shape[0]):
        output_temp = downsampled_output[i]
        label_temp = downsampled_label[i]
        output_temp = output_temp.view(output_temp.shape[0], output_temp.shape[1] * output_temp.shape[2]).permute(1, 0)
        label_temp = label_temp.reshape(label_temp.shape[0], label_temp.shape[1] * label_temp.shape[2]).permute(1, 0)
        # output_temp = output_temp.view(output_temp.shape[1]*output_temp.shape[2], output_temp.shape[0])
        # label_temp = label_temp.reshape(label_temp.shape[1]*label_temp.shape[2], label_temp.shape[0])

        with torch.no_grad():
            C = torch.cdist(output_temp, label_temp, p=2.0) ** 2
            gamma1 = ot.emd(ot.unif(output_temp.shape[0]),
                            ot.unif(label_temp.shape[0]),
                            C.cpu().squeeze().numpy())
            gamma1 = torch.tensor(gamma1)
        # mask_temp = [i for i in range(gamma1.shape[0]) if gamma1[i, i] == gamma1[i, :].max()]  # 提取最大值在对角线上的序号
        mask_temp = [i for i in range(gamma1.shape[0]) if i == torch.argmax(gamma1[i, :])]
        # if len(mask_temp) > 5:
        #     print('CA length: ', len(mask_temp))
        # 掩膜处理
        new_vector = torch.zeros(1, gamma1.shape[0])
        for index in mask_temp:
            new_vector[0, index] = 1
        new_vector = new_vector.view(1, 1, downsampled_label.shape[2], downsampled_label.shape[3])  # 向量变成图像
        upsampled_tensor = torch.nn.functional.interpolate(new_vector, size=(output.shape[2], output.shape[2]), mode='bilinear',align_corners=False)

        # 假设upsampled_tensor是需要修改的张量，包含0和其他数值，其他数值变成1
        modified_tensor = (upsampled_tensor != 0).float()
        mask[i,:,:,:] = modified_tensor

    return mask.squeeze(1)

def class_diff(fmap, pred_mask, m1, m2, p1L, p1H):
    total_max = fmap.permute(1, 0, 2, 3) * pred_mask  # cls l的集中特征
    cal_data1 = (total_max * m1)  # .cpu().numpy()  CA（cls l）
    cal_data2 = (total_max * m2)  # .cpu().numpy()  VA（cls l）
    mVA = p1L * (torch.mean(cal_data2[cal_data2 > 0]) - torch.mean(total_max[total_max > 0])) ** 2
    mCA = p1H * (torch.mean(cal_data1[cal_data1 > 0]) - torch.mean(total_max[total_max > 0])) ** 2
    return mCA + mVA


class Timer():
    '''A wrapper class for printing out what is running and how long it took.
    
    Use as:
    ```
    with utils.Timer("running stuff"):
        # do stuff
    ```

    This will output:
    ```
    Starting 'running stuff'
    # any output from 'running stuff'
    Finished 'running stuff' in 12.45 seconds
    ```
    '''

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.tic = float(time.time())
        print("Starting '%s'" % (self.message))

    def __exit__(self, type, value, traceback):
        print("Finished '%s' in %0.4f seconds" % (self.message, time.time() - self.tic))


def fit(model, device, data_loader, num_batches, optimizer, criterion, epoch, dataloader_val, fa_i, memo=''):
    model.train()

    losses = []
    tic = time.time()
    class_num = len(NLCD_CLASSES)  # the class_num could be set according to the dataset.
    class_std_max1 = torch.zeros(class_num).to(device)
    class_std_avg1 = torch.zeros(class_num).to(device)
    weighted_loss = WeightedLoss(ignore_lb=4)

    target0 = 0.0; target1 = 0.0; target2 = 0.0; target3 = 0.0
    # for batch_idx, (data, targets) in enumerate(data_loader):
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        if data.shape[0] < 2:
            break
            # data = torch.cat([data, data],dim=0)
            # targets = torch.cat([targets, targets],dim=0)
        model.train()
        data = data.to(device)  # b, 4, H, W
        targets = targets.to(device)  # b, H, W
        optimizer.zero_grad()
        outputs = model(data)  # b, cls, H, W
        Foutputs = F.softmax(outputs, dim=1).cpu()
        maxoutput = torch.max(Foutputs, axis=1)
        weights_for_loss = maxoutput[0]
        # Tmean = torch.mean(maxoutput[0])
        # criterion = nn.CrossEntropyLoss(weight=Tmean)
        # 基于OT的置信区选择
        mask1 = ot_mask(Foutputs, targets).to(torch.long).to(device)  ## for calculating the CAS mask
        mask2 = torch.where(mask1 == 1, 0, 1).to(torch.long).to(device)  ## mask2 represents the VA
        for j in range(class_num):
            pred_mask = torch.where(maxoutput[1] == j + 1, 1, 0).to(torch.long).to(device)
            class_pix_num = len(pred_mask[pred_mask == 1])
            part1 = pred_mask * mask1
            HC_pix_num = len(part1[part1 == 1])
            if class_pix_num == 0:
                p1H = 0
                p1L = 0
            else:
                p1H = HC_pix_num / class_pix_num
                p1L = 1 - p1H

            class_std_max1[j] = class_diff(model.featuremap_max, pred_mask, mask1, mask2, p1L, p1H)
            class_std_avg1[j] = class_diff(model.featuremap_avg, pred_mask, mask1, mask2, p1L, p1H)

        class_std_loss_max1 = torch.nansum(class_std_max1)
        class_std_loss_avg1 = torch.nansum(class_std_avg1)

        total_std_loss = (class_std_loss_max1 + class_std_loss_avg1) / 2  # 无监督损失

        # outputs = (outputs.permute(1, 0, 2, 3) * mask1).permute(1, 0, 2, 3)
    #  add the weak-supervised CE loss and the DVA loss，0.01 is the paramater（gamma）
        if mask1[mask1 > 0].shape[0] == 0:
            continue

        for i_b in range(targets.shape[0]):
            targets[i_b, :][mask1[i_b, :] == 0] = 4
        targets = targets.long()

        target0 = target0 + targets[targets == 0].shape[0]
        target1 = target1 + targets[targets == 1].shape[0]
        target2 = target2 + targets[targets == 2].shape[0]
        target3 = target3 + targets[targets == 3].shape[0]

        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0]) # torch.tensor([10.0, 1.0, 2.5, 10.0])
        # criterion_temp = nn.CrossEntropyLoss(weight=weights, ignore_index=4).to(targets.device)
        #
        # loss = criterion_temp(outputs, targets) + 0.1 * total_std_loss
        loss = weighted_loss(outputs, targets, weights_for_loss) + 0.1 * total_std_loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx != 0 and batch_idx % 1500 == 0:
            print(torch.mean(torch.tensor(losses)), "; ", target0, "; ", target1, "; ", target2, "; ", target3)
            target0 = 0.0
            target1 = 0.0
            target2 = 0.0
            target3 = 0.0
            model.eval()
            model.to(device)
            num_classes = 4
            scores = metrics.RunningScoreForTensorEnhanced(num_classes)
            for i_iter, batch in enumerate(dataloader_val):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(images)
                    outputs = outputs.max(1)[1]
                scores.update(labels.data, outputs)
            val_iu = scores.get_scores()
            # fa_i = open(output_dir + strPathname + "m_accuracy_values_b.txt", 'w')
            fa_i.write("________________________________________________________________________________" + "\n")
            print(str(epoch) + ": " + str(batch_idx) + " val mIoU: ", val_iu[0]["Mean IoU"])
            print(str(epoch) + ": " + str(batch_idx) + " val OA: ", val_iu[0]["Overall Acc"])
            fa_i.write(str(epoch) + ": " + str(batch_idx) + '\t' + "val MIOU" + '\t' + str(val_iu[0]["Mean IoU"]) + '\n')
            fa_i.write(str(epoch) + ": " + str(batch_idx) + '\t' + "val OA" + '\t' + str(val_iu[0]["Overall Acc"]) + '\n')
            fa_i.write(str(epoch) + ": " + str(batch_idx) + '\t' + "Kappa" + '\t' + str(val_iu[0]["Kappa"]) + '\n')
            fa_i.write(str(epoch) + ": " + str(batch_idx) + '\t' + "fwavacc" + '\t' + str(val_iu[0]["FreqW Acc"]) + '\n')
            fa_i.write("________________________________________________________________________________" + "\n")
            for i in range(num_classes):
                fa_i.write("IOU Class " + str(i) + ": " + str(val_iu[1][i]) + "\n")
                print("IOU Class " + str(i) + ": ", val_iu[1][i])
            fa_i.write("________________________________________________________________________________" + "\n")
            for i in range(num_classes):
                fa_i.write("P Class " + str(i) + ": " + str(val_iu[2][i]) + "\n")
                print("P Class " + str(i) + ": ", val_iu[2][i])
            fa_i.write("________________________________________________________________________________" + "\n")
            for i in range(num_classes):
                fa_i.write("R Class " + str(i) + ": " + str(val_iu[3][i]) + "\n")
                print("R Class " + str(i) + ": ", val_iu[3][i])
            fa_i.write("________________________________________________________________________________" + "\n")
            for i in range(num_classes):
                fa_i.write("F1 Class " + str(i) + ": " + str(val_iu[4][i]) + "\n")
                print("F1 Class " + str(i) + ": ", val_iu[4][i])
            fa_i.write("________________________________________________________________________________" + "\n")

        model.train()

    avg_loss = np.mean(losses)
    print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time() - tic, avg_loss), end=""
    )
    print("")

    return [avg_loss]


def validation(model, device, data_loader, num_class, epoch, fa_i):
    model.eval()
    model.to(device)
    num_classes = num_class
    scores = metrics.RunningScoreForTensorEnhanced(num_classes)
    for i_iter, batch in enumerate(data_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            outputs = outputs.max(1)[1]
        scores.update(labels.data, outputs)
    val_iu = scores.get_scores()
    # fa_i = open(output_dir + strPathname + "m_accuracy_values_b.txt", 'w')
    fa_i.write("________________________________________________________________________________" + "\n")
    print(str(epoch) + " val mIoU: ", val_iu[0]["Mean IoU"])
    print(str(epoch) + " val OA: ", val_iu[0]["Overall Acc"])
    fa_i.write(str(epoch) + '\t' + "val MIOU" + '\t' + str(val_iu[0]["Mean IoU"]) + '\n')
    fa_i.write(str(epoch) + '\t' + "val OA" + '\t' + str(val_iu[0]["Overall Acc"]) + '\n')
    fa_i.write(str(epoch) + '\t' + "Kappa" + '\t' + str(val_iu[0]["Kappa"]) + '\n')
    fa_i.write(str(epoch) + '\t' + "fwavacc" + '\t' + str(val_iu[0]["FreqW Acc"]) + '\n')
    fa_i.write("________________________________________________________________________________" + "\n")
    for i in range(num_classes):
        fa_i.write("IOU Class " + str(i) + ": " + str(val_iu[1][i]) + "\n")
        print("IOU Class " + str(i) + ": ", val_iu[1][i])
    fa_i.write("________________________________________________________________________________" + "\n")
    for i in range(num_classes):
        fa_i.write("P Class " + str(i) + ": " + str(val_iu[2][i]) + "\n")
        print("P Class " + str(i) + ": ", val_iu[2][i])
    fa_i.write("________________________________________________________________________________" + "\n")
    for i in range(num_classes):
        fa_i.write("R Class " + str(i) + ": " + str(val_iu[3][i]) + "\n")
        print("R Class " + str(i) + ": ", val_iu[3][i])
    fa_i.write("________________________________________________________________________________" + "\n")
    for i in range(num_classes):
        fa_i.write("F1 Class " + str(i) + ": " + str(val_iu[4][i]) + "\n")
        print("F1 Class " + str(i) + ": ", val_iu[4][i])
    fa_i.write("________________________________________________________________________________" + "\n")


def evaluate(model, device, data_loader, num_batches, criterion, epoch, memo=''):
    model.eval()
    model.to(device)
    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    avg_loss = np.mean(losses)

    print('[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time() - tic, avg_loss), end=""
    )
    print("")

    return [avg_loss]


def score(model, device, data_loader, num_batches):
    model.eval()

    num_classes = model.module.segmentation_head[0].out_channels
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        batch_size = data.shape[0]
        predictions[idx:idx + batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WeightedLoss(nn.Module):
    def __init__(self, ignore_lb=255):
        super(WeightedLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels, weights):
        loss = self.criteria(logits, labels)
        weights = weights.to(logits.device)
        # weights = torch.exp(weights-torch.mean(weights))
        loss = (loss * torch.exp(weights)).view(-1)
        loss = loss[torch.nonzero(loss)]

        return torch.mean(loss)

