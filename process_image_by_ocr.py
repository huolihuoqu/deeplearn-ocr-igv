import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import os
import numpy as np
import cv2
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ocr = PaddleOCR(use_angle_cls=True, lang="en", det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3',
                det_limit_side_len=2000)  # need to run only once to download and load model into memory

def process_images(input_folder_path, output_folder_path):
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 确保文件是图片文件
            # chrY_31054326-31057224_gt01_ins_true_2415.png
            # 只处理 ins 类型变异
            if filename.split('_')[3] == 'del':
                continue
            # 构建输入文件的完整路径
            input_file_path = os.path.join(input_folder_path, filename)

            # 读取图像
            image = cv2.imread(input_file_path)

            # 定义白色（BGR 格式）
            lower_white = np.array([220, 0, 150])
            upper_white = np.array([255, 255, 255])

            # 创建一个掩模，用于标记白色区域
            white_mask = cv2.inRange(image, lower_white, upper_white)

            # 将白色区域变为黑色
            image[white_mask > 0] = [0, 0, 0]  # 黑色 (BGR格式)

            # *****************************************************************************
            # 定义紫色（BGR 格式）
            lower_purple = np.array([200, 0, 100])
            upper_purple = np.array([255, 200, 200])

            # 创建一个掩模，用于标记紫色区域
            purple_mask = cv2.inRange(image, lower_purple, upper_purple)

            # 将紫色区域变为白色
            image[purple_mask > 0] = [255, 255, 255]  # 白色 (BGR格式)

            # ******************************************************************************
            # 定义灰色
            lower_gray = np.array([200, 200, 200])
            upper_gray = np.array([205, 205, 205])

            # 创建一个掩模，用于标记灰色区域
            gray_mask = cv2.inRange(image, lower_gray, upper_gray)

            # 将灰色区域变为白色
            image[gray_mask > 0] = [255, 255, 255]  # 白色 (BGR格式)

            # *****************************************************************************
            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder_path, filename)

            cv2.imwrite(output_file_path, image)

    return "Images processed successfully"


def ocr_and_mark(input_image_path, output_image_path, a = 0.1):
    for filename in os.listdir(input_image_path):
        # 构建输入文件的完整路径
        img_path = os.path.join(input_image_path, filename)
        data = {}
        # Paddleocr 目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改 lang 参数进行切换
        # 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
        # img_path = 'igv_img/chr1_505510-507136_gt11_ins_false_1355.png'
        data['filename'] = img_path.split('\\')[-1]
        l = int(img_path.split('_')[-1].split('.')[0])
        low = l * (1-a)
        high = l * (1+a)
        # img = cv2.imread(img_path)
        result = ocr.ocr(img_path, cls=True)
        vertices = []

        # 使用 ppocr 获取指定变异长度的坐标位置
        if len(result) > 0 and result[0] != None:
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    digit = line[-1][0]
                    # 判断条件，这里以数字长度为例
                    # 提取数字部分
                    if digit == None or digit == '': continue
                    digit = (''.join(filter(str.isdigit, digit)))
                    if digit == None or digit == '': continue
                    digit = int(digit)
                    if digit >= low and digit <= high:
                        # 获取文本框坐标
                        vertices.append(np.array(line[0]).reshape((-1, 1, 2)).astype(np.int32))
            data['pos'] = vertices

        # 对获取到的位置进行 mark
        color=(0, 255, 0)
        file_name = data.get('filename')
        input_file = os.path.join(input_image_path, file_name)
        out_file = os.path.join(output_image_path, file_name)
        img = cv2.imread(input_file)
        if 'pos' not in data.keys():
            cv2.imwrite(out_file, img)
        for pos in data.get('pos'):
            cv2.fillPoly(img, [pos], color)
            cv2.imwrite(out_file, img)

    return "Images ocr and mark successfully"


def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Sub-parser for process_images
    parser_process = subparsers.add_parser('process_images', help='Process images in a folder.')
    parser_process.add_argument('input_folder_path', type=str, help='Input folder path containing images')
    parser_process.add_argument('output_folder_path', type=str, help='Output folder path for processed images')

    # Sub-parser for ocr_and_mark
    parser_ocr = subparsers.add_parser('ocr_and_mark', help='Perform OCR and mark on an image.')
    parser_ocr.add_argument('input_image_path', type=str, help='Input image path')
    parser_ocr.add_argument('output_image_path', type=str, help='Output image path')
    parser_ocr.add_argument('--a', type=float, default=0.1, help='Parameter a')

    args = parser.parse_args()

    if args.command == 'process_images':
        result = process_images(args.input_folder_path, args.output_folder_path)
        print(result)
    elif args.command == 'ocr_and_mark':
        result = ocr_and_mark(args.input_image_path, args.output_image_path, args.a)
        print(result)

# 调用 process_images 函数
# python process_image_by_ocr.py process_images input_folder output_folder

# 调用 ocr_and_mark 函数
# python process_image_by_ocr.py ocr_and_mark input_image_path output_image_path --a 0.1

if __name__ == "__main__":
    main()