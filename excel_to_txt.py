import pandas as pd

import pandas as pd
import argparse
# excel 中需要包含以下字段：‘#CHROM’，‘POS’，‘INFO'，‘GT’
# 由 INFO 中提取出 SVLEN、SVTYPE
# a 为偏移系数
def update_excel(inputFile, outputFile, a = 0.1):
    try:
        # 读取Excel文件
        df = pd.read_excel(inputFile)

        # 检查是否包含必要的列
        required_columns = ['#CHROM', 'POS', 'INFO', 'GT']
        for col in required_columns:
            if col not in df.columns:
                return f"Missing required column: {col}"

        # 从'INFO'列中提取所需信息
        df['END'] = df['INFO'].str.extract(r'END=(\d+);')
        df['SVTYPE'] = df['INFO'].str.extract(r'SVTYPE=([^;]+);')
        df['SVLEN'] = df['INFO'].str.extract(r'SVLEN=(-?\d+);')

        # 新增 START 列
        df['START'] = df['END'] - (df['SVLEN'].abs()).astype(int)

        # 新增 NEWSTART 列
        df['NEWSTART'] = df['START'] - (df['SVLEN'].abs() * a).astype(int)

        # 新增 NEWEND 列
        df['NEWEND'] = df['END'] + (df['SVLEN'].abs() * a).astype(int)

        df['TRUEorFALSE'] = 'noKnow'

        # 处理数据，按需修改列名
        df['#CHROM'] = df['#CHROM'].astype(str)
        df['NEWSTART'] = df['NEWSTART'].astype(str)
        df['NEWEND'] = df['NEWEND'].astype(str)

        # chrY_31054326-31057224_gt01_ins_true_2415.png
        df['FILENAME'] = (
                'chr' +
                df['#CHROM'] + '_' +
                df['NEWSTART'] + '-' + df['NEWEND'] + '_' +
                df['GT'].str.lower() + '_' +
                df['SVTYPE'].str.lower() + '_' +
                df['TRUEorFALSE'].astype(str).str.lower() + '_' +
                df['SVLEN'].abs().astype(int).astype(str)
        )

        # 将修改后的DataFrame写回Excel文件
        df.to_excel(outputFile, index=False)

        return f"Data successfully written to {outputFile}"
    except Exception as e:
        return str(e)

def generate_igv_script(excel_file, bam_file, image_output, txt_file):
    # excel 中需要包含以下字段：‘#CHROM’，‘POS’，‘SVLEN'，'SVTYPE',‘GT’
    # 由 INFO 中提取出 SVLEN、SVTYPE
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file)

        # 初始字符串
        string = 'new\n' + 'genome hg19\n' + 'load ' + bam_file + '\n' + 'snapshotDirectory ' + image_output + '\n'

        # 创建一个新列，以循环的形式逐行读取和处理数据
        for index, row in df.iterrows():
            # 在每一行的数据前后拼接 'prefix\n' 和 '\nsuffix'
            goto = 'goto ' + 'chr' + str(row["#CHROM"]) + ':' + \
                   str(row["NEWSTART"]) + '-' + str(row["NEWEND"]) + '\n'
            mid_str = 'sort position\nexpend\n'
            file = 'snapshot ' + row["FILENAME"] + '.png\n'

            # 将结果写入相应的位置
            string += goto + mid_str + file

        # 将字符串写入文本文件
        with open(txt_file, "w") as f:
            f.write(string)

        return "Script successfully written to " + txt_file

    except Exception as e:
        return str(e)

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Sub-parser for update_excel
    parser_update = subparsers.add_parser('update_excel', help='Update an Excel file.')
    parser_update.add_argument('inputFile', type=str, help='Input Excel file')
    parser_update.add_argument('outputFile', type=str, help='Output Excel file')
    parser_update.add_argument('--a', type=float, default=0.1, help='Parameter a')

    # Sub-parser for generate_igv_script
    parser_generate = subparsers.add_parser('generate_igv_script', help='Generate an IGV script.')
    parser_generate.add_argument('excel_file', type=str, help='Input Excel file')
    parser_generate.add_argument('bam_file', type=str, help='BAM file')
    parser_generate.add_argument('image_output', type=str, help='Image output directory')
    parser_generate.add_argument('txt_file', type=str, help='Output text file')

    args = parser.parse_args()

    if args.command == 'update_excel':
        result = update_excel(args.inputFile, args.outputFile, args.a)
        print(result)
    elif args.command == 'generate_igv_script':
        result = generate_igv_script(args.excel_file, args.bam_file, args.image_output, args.txt_file)
        print(result)


# python excel_to_txt.py update_excel input.xlsx output.xlsx --a 0.2
# 调用 generate_igv_script 函数
# python excel_to_txt.py generate_igv_script input.xlsx input.bam output_dir output.txt

if __name__ == "__main__":
    main()