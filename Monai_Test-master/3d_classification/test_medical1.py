import logging
import os
import sys
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    SpatialPad,
    RandFlip,
    RandZoom,
    AsDiscrete,
)
import math
from monai.metrics import ConfusionMatrixMetric
from sklearn.model_selection import KFold
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import time
import torch.multiprocessing
from utils import get_model, SmoothCrossEntropyLoss, draw_confusion_graph, FocalLossCrossEntropyLoss, draw_auc_graph

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='input args')

    parser.add_argument('-d', '--data', type=str, required=False, default="E:/KTH/breast/data/tdsc_crop3_100",
                        help='data directory')
    parser.add_argument('-l', '--label', type=str, required=False, default='E:/KTH/breast/data/QIN-Breast_TreatmentResponse。xlsx',
                        help='label csv')
    parser.add_argument('-rs', '--resize', type=int, required=False, default=128, help='resize size')
    parser.add_argument('-ps', '--padsize', type=int, required=False, default=128, help='Pad size')
    parser.add_argument('-lr', '--learningrate', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-ep', '--epochs', type=int, required=False, default=60, help="epochs")
    parser.add_argument('-da', '--dataaugmentation', type=int, required=False, default=15, help="data augmentation")
    parser.add_argument('-dr', '--dropout', type=int, required=False, default=0.5, help="dropout rate")
    parser.add_argument('-m', '--model', type=str, required=False, default='densenet', help='model')

    args = parser.parse_args()

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # create new fold
    current_directory = os.getcwd()
    args_directory = str(args.resize) + "_" + str(args.learningrate) + "_" \
                     + str(args.dataaugmentation) + "_" + str(args.dropout) + '_' + str(args.model) + '_data1' + '_fortest1'
    output_directory = os.path.join(current_directory, "results2", args_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # load image
    directory = args.data  # data directory
    images = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.nii')]
    images = np.array(images)

    # load label
    df = pd.read_csv(args.label)  # label directory
    labels = df['label']
    labels.replace({'pCR': 0, 'non-pCR': 1}, inplace=True)
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    labels = np.array(labels)

    # transform
    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(),
         # RandRotate90(prob=0.1),  # 10%的概率随机旋转90度
         RandFlip(spatial_axis=0, prob=0.1),  # 10%的概率进行随机翻转
         RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.1),  # 10%的概率进行随机缩放
         SpatialPad((args.padsize, args.padsize, args.padsize), mode='constant'),
         Resize((args.resize, args.resize, args.resize))])

    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(),
         SpatialPad((args.padsize, args.padsize, args.padsize), mode='constant'),
         Resize((args.resize, args.resize, args.resize))])

    # data to store result
    history_list = []

    # 5 cross val test
    kf = KFold(n_splits=5)

    # begin train and test
    for i, (train_index, val_index) in enumerate(kf.split(images)):
        start_time = time.perf_counter()

        print("-" * 10, f"{i} fold", "-" * 10)

        # create a validation data loader
        train_fold_images, val_fold_images = images[train_index], images[val_index]
        train_fold_labels, val_fold_labels = labels[train_index], labels[val_index]
        val_ds = ImageDataset(image_files=val_fold_images, labels=val_fold_labels, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=5, num_workers=6, pin_memory=pin_memory)

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer set loss function and optimizer
        model = get_model(args).to(device)
        loss_function = SmoothCrossEntropyLoss(label_smoothing=0.5)
        # loss_function = FocalLossCrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), args.learningrate)

        best_metric = -1
        best_metric_epoch = -1
        best_auc = -1
        best_auc_epoch = -1
        best_model = {'model': [],
                      'val_label': [],
                      'val_pre': [],
                      'value': [],
                      'epoch': []}
        step = 0
        metric_values = []
        history_figure = {'val_auc': [],
                          'val_acc': [],
                          'val_loss': [],
                          'train_acc': [],
                          'train_loss': []}

        max_epochs = args.epochs

        # start one epoch
        for epoch in range(max_epochs):

            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")

            model.train()

            epoch_loss = 0
            val_epoch_loss = 0
            num_correct = 0.0
            metric_count = 0
            step += 1
            data_step = 0
            val_data_step = 0

            # start one augmentation batch
            for j in range(args.dataaugmentation):

                # create a training data loader
                train_ds = ImageDataset(image_files=train_fold_images, labels=train_fold_labels,
                                        transform=train_transforms)
                train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=6, pin_memory=pin_memory)

                epoch_len = len(train_ds) // train_loader.batch_size
                epoch_max = epoch_len * args.dataaugmentation

                # start one batch
                for batch_data in train_loader:

                    # start a train
                    inputs_cuda, labels_cuda = batch_data[0].to(device), batch_data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs_cuda)

                    # get acc score
                    value = torch.eq(outputs.argmax(dim=1), labels_cuda.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

                    # get loss value and count
                    loss = loss_function(outputs, labels_cuda)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    # show step result
                    data_step += 1
                    print(f"{data_step}/{epoch_max}/{step}/{max_epochs}, train_loss: {loss.item():.4f}")

            metric = num_correct / metric_count
            history_figure['train_acc'].append(metric)

            epoch_loss /= data_step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            history_figure['train_loss'].append(epoch_loss)

            model.eval()  # 进入评估模式

            all_preds = []
            all_labels = []

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()
                    val_data_step += 1

                    all_preds_binary = (val_outputs > 0.5).float()
                    all_preds.append(all_preds_binary.detach())
                    all_labels.append(val_labels.detach())

                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            val_epoch_loss /= val_data_step
            history_figure['val_loss'].append(val_epoch_loss)

            # compute AUC and find best AUC
            auc = monai.metrics.compute_roc_auc(all_labels, all_preds)
            history_figure['val_auc'].append(auc)
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_directory, f"best_metric_model_{i}.pth"))

            # compute ACC and find best ACC
            metric = num_correct / metric_count
            metric_values.append(metric)
            history_figure['val_acc'].append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

            # find best model and record 5 best model
            if (len(best_model['model']) < 5 or auc >= min(best_model['value'])) and math.isnan(auc)==False:

                print('try to write value')

                if len(best_model['model']) == 5:
                    print('exchange occured:')
                    worst_index = best_model['value'].index(min(best_model['value']))
                    print(f'pop value: {best_model["value"][worst_index]}')
                    print(f'pop epoch: {best_model["epoch"][worst_index]}')
                    best_model['model'].pop(worst_index)
                    best_model['value'].pop(worst_index)
                    best_model['val_pre'].pop(worst_index)
                    best_model['val_label'].pop(worst_index)
                    best_model['epoch'].pop(worst_index)

                best_model['model'].append(model)
                best_model['value'].append(auc)
                best_model['val_pre'].append(all_preds)
                best_model['val_label'].append(all_labels)
                best_model['epoch'].append(epoch)

            # show epoch result
            print(f"Current epoch: {epoch + 1} current auc_score: {auc:.4f}")
            print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
            print(f"Best auc_score: {best_auc:.4f} at epoch {best_auc_epoch}")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

        # print train loss value figure
        plt.figure()
        plt.plot(range(max_epochs), history_figure['train_loss'], label='train_loss')
        plt.plot(range(max_epochs), history_figure['val_loss'], label='val_loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_directory, f"fold{i}_loss_figure.png"))

        # print acc value figure
        plt.figure()
        plt.plot(range(max_epochs), history_figure['train_acc'], label='train_acc')
        plt.plot(range(max_epochs), history_figure['val_acc'], label='val_acc')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_directory, f"fold{i}_acc_figure.png"))

        # print auc value figure
        plt.figure()
        plt.plot(range(max_epochs), history_figure['val_acc'], label='val_acc')
        plt.plot(range(max_epochs), history_figure['val_auc'], label='val_auc')
        plt.title('AUC')
        plt.legend()
        plt.savefig(os.path.join(output_directory, f"fold{i}_auc_figure.png"))

        fold_directory = f'fold{i}'
        output_directory_fold = os.path.join(output_directory, fold_directory)
        if not os.path.exists(output_directory_fold):
            os.makedirs(output_directory_fold)

        # save best five model, confusion graph
        for i_model, model_item in enumerate(best_model['model']):
            torch.save(model_item.state_dict(), os.path.join(output_directory_fold, f'best_model_{i_model}.pt'))
            draw_confusion_graph(y=best_model['val_label'][i_model], y_pred=best_model['val_pre'][i_model],
                                 directory=os.path.join(output_directory_fold, f'confusion_graph_{i_model}.png'))
            draw_auc_graph(y=best_model['val_label'][i_model], y_pred=best_model['val_pre'][i_model],
                                 directory=os.path.join(output_directory_fold, f'AUC_graph_{i_model}.png'))
            print(best_model['epoch'][i_model])

        # record best auc and acc
        history = {
            'num_fold': i,
            'best_auc': best_auc,
            'best_acc': best_metric,
        }
        history_list.append(history)

        print(f"Fold{i} Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Time used per fold: {elapsed_time} seconds")
        # writer.close()

    # print best auc and acc for each fold
    with open(os.path.join(output_directory, "training_history_folds.txt"), 'w') as file:
        file.write(str(history_list))


if __name__ == "__main__":
    main()