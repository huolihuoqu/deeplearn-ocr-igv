# deeplearn-ocr-igv

## Introduction

This work batch determines true and false positive variants in structural variants in IGV images, using a multi-task learning approach to simultaneously detect homozygous and heterozygous variants.

## Usage

### Prepraration

```shell
git clone https://github.com/huolihuoqu/deeplearn-ocr-igv.git
cd deeplearn-ocr-igv
pip install -r requirements.txt
```

### Processing Excel To Txt

```
python update_excel.py \
  --inputFile <path> --outputFile <outPath> \
  --a <0.1>
```

#### Options:

- --inputFile: The input Excel file path needs to contain the following fields: '#CHROM', 'POS', 'INFO', 'GT'.
- --outputFile:  Output path of the processed Excel file.
- --a:  The offset size determines the extension of the variant length before and after when using IGV to generate images.

#### For Example:

```python
python excel_to_txt.py update_excel yourInput.xlsx yourOutput.xlsx 0.1
```

```
python generate_igv_script.py \
  --excel_file <path> --bam_file <path> --image_output <directory> --txt_file <path>
```

#### Options:

- --excel_file: The excel path after the previous operation.
- --bam_file:  The path to the bam file required when generating images using IGV.
- --image_output:  Image storage directory when using IGV to generate images.
- --txt_file:  When using scripts in conjunction with IGV to generate images, the script storage location.

#### For Example:

```
python excel_to_txt.py generate_igv_script input.xlsx input.bam output_img_dir script.txt
```

### Process and OCR Image

```
python process_images.py \
  --input_folder_path <path> --output_folder_path <outPath>

python ocr_and_mark \
  --input_image_path <path> -- output_image_path <outPath> \
  --a <0.1>
```

#### For Example:

```
python process_image_by_ocr.py process_images input_folder output_folder
# 调用 ocr_and_mark 函数
python process_image_by_ocr.py ocr_and_mark input_image_path output_image_path --a 0.1
```

### Test Image

```
python test.py \
  --image_path <path> --excel_path <path>
```

#### For Example:

```
python test.py test imagePath excelPath
```

### Result

The recognition results of each image are saved in the final Excel.