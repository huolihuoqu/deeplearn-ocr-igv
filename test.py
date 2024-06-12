import torch.nn as nn
import torchvision
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from utils import get_score, matplot_auc, matplot_recall, matplot_prec, matplot_acc, matplot_loss, matplot_F1
from models import MultiTaskDataset, AttentionParallelNet
import pandas as pd
import argparse
def test(images_path, excel_path):

    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    val_dataset = MultiTaskDataset(images_path, transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    print(len(val_dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object
    # 四个 loss
    homIsPositive_loss = nn.CrossEntropyLoss()  # Includes Softmax
    homType_loss = nn.CrossEntropyLoss()  # include Softmax
    hetIsPositive_loss = nn.CrossEntropyLoss()
    hetType_loss = nn.CrossEntropyLoss()
    model = AttentionParallelNet(4).to(device=device)
    model.load_state_dict(torch.load(r'./best_model/attParallel_best.pth'))
    model.eval()

    # 验证 ********************************************************************************************************
    # ************************************************************************************************************
    isPositive_acc_val = [[],[]]
    type_acc_val = [[],[]]
    isPositive_loss_val = [[],[]]
    type_loss_val = [[],[]]
    all_loss_val = []

    isPositive_prec_val = [[],[]]
    isPositive_recall_val = [[],[]]
    isPositive_f1_val = [[],[]]
    isPositive_auc_val = [[],[]]

    epoch = 1

    data = {}

    if True:  # Simplified the condition
        epoch_isPositive_true_label_1 = []
        epoch_isPositive_true_label_3 = []
        epoch_type_true_label_1 = []
        epoch_type_true_label_3 = []

        epoch_isPositive_prec_1 = []
        epoch_isPositive_prec_3 = []
        epoch_type_prec_1 = []
        epoch_type_prec_3 = []

        epoch_isPositive_prec_class_1 = []
        epoch_isPositive_prec_class_3 = []
        epoch_type_prec_class_1 = []
        epoch_type_prec_class_3 = []

        total_training_loss = 0
        task13_loss = 0

        oneIsPositive_total_training_loss = 0
        oneType_total_training_loss = 0
        threeIsPositive_total_training_loss = 0
        threeType_total_training_loss = 0

        with torch.no_grad():
            for i, data_point in enumerate(val_dataloader):
                inputs = data_point["image"].to(device=device)
                isPositive_label = data_point["isPositive"].to(device=device)
                type_label = data_point["type"].to(device=device)
                filenames = data_point["filename"]  # Assuming each data point has a 'filename' key

                oneIsPositive_output, oneType_output, threeIsPositive_output, threeType_output = model(inputs)

                isPositive_mask1 = isPositive_label.eq(0) | isPositive_label.eq(1)
                isPositive_mask3 = isPositive_label.eq(2) | isPositive_label.eq(3)

                type_mask1 = type_label.eq(0) | type_label.eq(1)
                type_mask3 = type_label.eq(2) | type_label.eq(3)

                isPositive_loss_1 = 0
                isPositive_loss_3 = 0
                type_loss_1 = 0
                type_loss_3 = 0

                # Task 1 (hom true/false)
                _, oneIsPositive_prec_class = torch.max(oneIsPositive_output[isPositive_mask1], 1)
                isPositive_loss_1 = homIsPositive_loss(oneIsPositive_output[isPositive_mask1], isPositive_label[isPositive_mask1])
                epoch_isPositive_true_label_1.extend(isPositive_label[isPositive_mask1].data.tolist())
                epoch_isPositive_prec_1.extend(oneIsPositive_output[isPositive_mask1][:, 1].tolist())
                epoch_isPositive_prec_class_1.extend(oneIsPositive_prec_class.tolist())

                # Task 3 (het true/false)
                _, threeIsPositive_prec_class = torch.max(threeIsPositive_output[isPositive_mask3], 1)
                isPositive_loss_3 = hetIsPositive_loss(threeIsPositive_output[isPositive_mask3], isPositive_label[isPositive_mask3] - 2)
                epoch_isPositive_true_label_3.extend((isPositive_label[isPositive_mask3] - 2).data.tolist())
                epoch_isPositive_prec_3.extend(threeIsPositive_output[isPositive_mask3][:, 1].tolist())
                epoch_isPositive_prec_class_3.extend(threeIsPositive_prec_class.tolist())

                # Task 2 (hom type)
                _, oneType_prec_class = torch.max(oneType_output[type_mask1], 1)
                type_loss_1 = homType_loss(oneType_output[type_mask1], type_label[type_mask1])
                epoch_type_true_label_1.extend(type_label[type_mask1].data.tolist())
                epoch_type_prec_1.extend(oneType_output[type_mask1][:, 1].tolist())
                epoch_type_prec_class_1.extend(oneType_prec_class.tolist())

                # Task 4 (het type)
                _, threeType_prec_class = torch.max(threeType_output[type_mask3], 1)
                type_loss_3 = hetType_loss(threeType_output[type_mask3], type_label[type_mask3] - 2)
                epoch_type_true_label_3.extend((type_label[type_mask3] - 2).data.tolist())
                epoch_type_prec_3.extend(threeType_output[type_mask3][:, 1].tolist())
                epoch_type_prec_class_3.extend(threeType_prec_class.tolist())

                loss = isPositive_loss_1 + isPositive_loss_3 + 0.3 * type_loss_1 + 0.3 * type_loss_3
                task13_loss = isPositive_loss_1 + isPositive_loss_3

                oneIsPositive_total_training_loss += isPositive_loss_1
                oneType_total_training_loss += type_loss_1
                threeIsPositive_total_training_loss += isPositive_loss_3
                threeType_total_training_loss += type_loss_3
                total_training_loss += loss

                # Record predictions for each data point
                for idx in range(inputs.size(0)):
                    filename = filenames[idx]

                    data[filename] = {
                        "true_isPositive_label": isPositive_label[idx].item(),
                        "pred_isPositive_label_hom": 'N/A',
                        "pred_isPositive_label_het": 'N/A',
                        "true_type_label": type_label[idx].item(),
                        "pred_type_label_hom": 'N/A',
                        "pred_type_label_het": 'N/A',
                        "isPositive_prec_1": 'N/A',
                        "isPositive_prec_3": 'N/A',
                        "type_prec_1": 'N/A',
                        "type_prec_3": 'N/A'
                    }

                    if isPositive_mask1[idx]:
                        data[filename]["pred_isPositive_label_hom"] = oneIsPositive_prec_class[
                            isPositive_mask1.nonzero(as_tuple=True)[0].tolist().index(idx)].item()
                        # data[filename]["isPositive_prec_1"] = oneIsPositive_output[isPositive_mask1][isPositive_mask1.nonzero(as_tuple=True)[0].tolist().index(idx)][1].item()

                    if isPositive_mask3[idx]:
                        data[filename]["pred_isPositive_label_het"] = threeIsPositive_prec_class[
                            isPositive_mask3.nonzero(as_tuple=True)[0].tolist().index(idx)].item()
                        # data[filename]["isPositive_prec_3"] = threeIsPositive_output[isPositive_mask3][isPositive_mask3.nonzero(as_tuple=True)[0].tolist().index(idx)][1].item()

                    if type_mask1[idx]:
                        data[filename]["pred_type_label_hom"] = oneType_prec_class[
                            type_mask1.nonzero(as_tuple=True)[0].tolist().index(idx)].item()
                        # data[filename]["type_prec_1"] = oneType_output[type_mask1][type_mask1.nonzero(as_tuple=True)[0].tolist().index(idx)][1].item()

                    if type_mask3[idx]:
                        data[filename]["pred_type_label_het"] = threeType_prec_class[
                            type_mask3.nonzero(as_tuple=True)[0].tolist().index(idx)].item()
                        # data[filename]["type_prec_3"] = threeType_output[type_mask3][type_mask3.nonzero(as_tuple=True)[0].tolist().index(idx)][1].item()

            isPositive_loss_val[0].append(oneIsPositive_total_training_loss.data.cpu() / len(val_dataloader))
            isPositive_loss_val[1].append(threeIsPositive_total_training_loss.data.cpu() / len(val_dataloader))
            type_loss_val[0].append(oneType_total_training_loss.data.cpu() / len(val_dataloader))
            type_loss_val[1].append(threeType_total_training_loss.data.cpu() / len(val_dataloader))
            all_loss_val.append(total_training_loss.data.cpu() / len(val_dataloader))

            epoch_class_acc, epoch_score_acc, epoch_score_prec, epoch_score_recall, epoch_score_f1, epoch_score_auc = get_score(
                epoch_isPositive_true_label_1, epoch_isPositive_prec_1, epoch_isPositive_prec_class_1, type=2)
            isPositive_acc_val[0].append(epoch_score_acc)
            isPositive_prec_val[0].append(epoch_score_prec)
            isPositive_recall_val[0].append(epoch_score_recall)
            isPositive_f1_val[0].append(epoch_score_f1)
            isPositive_auc_val[0].append(epoch_score_auc)

            epoch_class_acc, epoch_score_acc, epoch_score_prec, epoch_score_recall, epoch_score_f1, epoch_score_auc = get_score(
                epoch_type_true_label_1, epoch_type_prec_1, epoch_type_prec_class_1, type=2)
            type_acc_val[0].append(epoch_score_acc)

            epoch_class_acc, epoch_score_acc, epoch_score_prec, epoch_score_recall, epoch_score_f1, epoch_score_auc = get_score(
                epoch_isPositive_true_label_3, epoch_isPositive_prec_3, epoch_isPositive_prec_class_3, type=2)
            isPositive_acc_val[1].append(epoch_score_acc)
            isPositive_prec_val[1].append(epoch_score_prec)
            isPositive_recall_val[1].append(epoch_score_recall)
            isPositive_f1_val[1].append(epoch_score_f1)
            isPositive_auc_val[1].append(epoch_score_auc)

            epoch_class_acc, epoch_score_acc, epoch_score_prec, epoch_score_recall, epoch_score_f1, epoch_score_auc = get_score(
                epoch_type_true_label_3, epoch_type_prec_3, epoch_type_prec_class_3, type=2)
            type_acc_val[1].append(epoch_score_acc)

        # 处理 data 将data中数据写入excel
        print('总计处理图片：{}'.format(len(data)))

        # 读取现有的Excel文件
        df_existing = pd.read_excel(excel_path)

        # 创建一个新的DataFrame存储新数据
        df_new_data = pd.DataFrame(columns=[
            'filename',
            'pred_isPositive_label',
            'pred_type_label'
        ])

        # 填充DataFrame
        for filename, values in data.items():
            pred_isPositive_label = values['pred_isPositive_label_het'] if values['pred_isPositive_label_het'] != 'N/A' else values['pred_isPositive_label_hom']
            pred_type_label = values['pred_type_label_het'] if values['pred_type_label_het'] != 'N/A' else values['pred_type_label_hom']

            df_new_data = df_new_data.append({
                'filename': filename,
                'pred_isPositive_label': pred_isPositive_label,
                'pred_type_label': pred_type_label
            }, ignore_index=True)

        # 将filename列重命名为FILENAME以匹配现有的Excel文件
        df_new_data.rename(columns={'filename': 'FILENAME'}, inplace=True)

        # 合并新数据到现有的DataFrame中，基于FILENAME列
        df_combined = df_existing.merge(df_new_data, on='FILENAME', how='left')

        # 写入更新后的DataFrame到新的Excel文件
        df_combined.to_excel(excel_path, index=False)

        print(f'Data has been written to {excel_path}')


def main():
    parser = argparse.ArgumentParser(description='Process images and update Excel file.')
    parser.add_argument('images_path', type=str, help='Path to the folder containing images.')
    parser.add_argument('excel_path', type=str, help='Path to the Excel file.')

    args = parser.parse_args()
    test(args.images_path, args.excel_path)

if __name__ == "__main__":
    main()