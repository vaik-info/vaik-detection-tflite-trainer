# vaik-detection-tflite-trainer

Train and export tflite model by own dataset.

## Example

## Install

```shell
pip install -r requirements.txt
```

## Usage

### Train

```shell
python train.py --train_image_dir_path ~/.vaik-mnist-detection-dataset/train \
                --train_label_dir_path ~/.vaik-mnist-detection-dataset/train \
                --valid_image_dir_path ~/.vaik-mnist-detection-dataset/valid \
                --valid_label_dir_path ~/.vaik-mnist-detection-dataset/valid \
                --classes_txt_path ~/.vaik-mnist-detection-dataset/classes.txt \
                --model_output_dir_path ~/output_model \
                --epoch_size 20 \
                --batch_size 8 \
                --max_detections 100
```

#### Output

![train](https://user-images.githubusercontent.com/116471878/200117986-96d5ad26-8905-4f4d-a1bb-d5754399c67b.png)

-----

### Export

```shell
python export.py --train_image_dir_path ~/.vaik-mnist-detection-dataset/train \
                --train_label_dir_path ~/.vaik-mnist-detection-dataset/train \
                --valid_image_dir_path ~/.vaik-mnist-detection-dataset/valid \
                --valid_label_dir_path ~/.vaik-mnist-detection-dataset/valid \
                --classes_txt_path ~/.vaik-mnist-detection-dataset/classes.txt \
                --model_output_dir_path ~/output_tflite_model \
                --epoch_size 20 \
                --batch_size 8 \
                --max_detections 100
```

### Export for coral

```shell
cd ~/output_tflite_model
edgetpu_compiler -s efficientdet-lite0.tflite
```

#### Output

![export](https://user-images.githubusercontent.com/116471878/200118102-2dd137b4-c39f-4b57-8a46-400b91275755.png)
