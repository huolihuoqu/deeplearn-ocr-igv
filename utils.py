import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
def get_score(true_data, result_prediction, result_class, type = 2):
    """
    分类问题的得分辅助函数
    :param true_data: 真实值，是要预测的目标
    :param result_prediction: 预测值，用模型预测出来的值
    :param save_roc_path: 保存roc曲线的路径
    :param model_name: 模型名
    :return: acc, prec, recall ,f1, roc_auc
    """
    types = type
    class_acc = []
    # print("true_data: ", true_data)
    # print("result_class: ", result_class)

    if types == 2:
        report_str = classification_report(true_data, result_class, labels=[0, 1])
        # 假设已经得到了 classification_report 函数的输出结果字符串 report_str
        report_list = report_str.splitlines()

        # 提取第二行到最后一行，即每个类别的行
        class_reports = report_list[2:types + 2]

        # 针对每个类别的行，将字符串分割成子列表
        class_reports = [row.split() for row in class_reports]

        # 将子列表中的字符串转换为相应的数据类型
        class_reports = [[float(num) if '.' in num else int(num) for num in row] for row in class_reports]
        # 输出每个类别的分类准确率
        class_acc = []
        for class_report in class_reports:
            # class_id = int(class_report[0])
            class_acc.append(class_report[1])
            # print(f"Class {class_id}: Accuracy = {class_report[1]}")

        acc = accuracy_score(true_data, result_class)  # 准确率
        prec = precision_score(true_data, result_class, average='micro')  # 精确率
        recall = recall_score(true_data, result_class, average='micro')  # 召回率
        f1 = f1_score(true_data, result_class, average='micro')  # F1

        fpr, tpr, thersholds = roc_curve(true_data, result_prediction)
        roc_auc = auc(fpr, tpr)
    else:
        acc = accuracy_score(true_data, result_class)  # 准确率
        prec = 0
        recall = 0
        f1 = 0

        # fpr, tpr, thersholds = roc_curve(true_data, result_prediction)
        # roc_auc = auc(fpr, tpr)
        roc_auc = 0

    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # if save_roc_path:
    #     plt.savefig(save_roc_path)
    # plt.show()
    score_list = [class_acc, acc, prec, recall, f1, roc_auc]
    # print('模型{}：'.format(model_name), score_list)
    return score_list

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc, true_acc_train, true_acc_val, false_acc_train, false_acc_val):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.plot(true_acc_train, label="true_train")
    plt.plot(true_acc_val, label="true_val")
    plt.plot(false_acc_train, label="false_train")
    plt.plot(false_acc_val, label="false_val")
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()


def matplot_prec(train_prec, val_prec):
    plt.plot(train_prec, label='train_prec')
    plt.plot(val_prec, label='val_prec')
    plt.legend(loc='best')
    plt.ylabel('prec')
    plt.xlabel('epoch')
    plt.title("训练集和验证集precision值对比图")
    plt.show()


def matplot_recall(train_recall, val_recall):
    plt.plot(train_recall, label='train_recall')
    plt.plot(val_recall, label='val_recall')
    plt.legend(loc='best')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.title("训练集和验证集recall值对比图")
    plt.show()


def matplot_F1(train_f1, val_f1):
    plt.plot(train_f1, label='train_f1')
    plt.plot(val_f1, label='val_f1')
    plt.legend(loc='best')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.title("训练集和验证集F1值对比图")
    plt.show()


def matplot_auc(train_auc, val_auc):
    plt.plot(train_auc, label='train_auc')
    plt.plot(val_auc, label='val_auc')
    plt.legend(loc='best')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.title("训练集和验证集AUC值对比图")
    plt.show()