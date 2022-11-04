import tensorflow as tf
from tflite_model_maker import object_detector
import os
import argparse


def read_classes_txt(classes_txt_path):
    label_map = {}
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]
    for index, a_class in enumerate(classes):
        label_map[index + 1] = a_class
    return label_map


def main(train_image_dir_path, train_label_dir_path, valid_image_dir_path, valid_label_dir_path,
                  classes_txt_path, ckpt_path, model_output_dir_path, max_detections):
    os.makedirs(model_output_dir_path, exist_ok=True)
    label_map = read_classes_txt(classes_txt_path)

    train_data_loader = object_detector.DataLoader.from_pascal_voc(images_dir=train_image_dir_path,
                                                                   annotations_dir=train_label_dir_path,
                                                                   label_map=label_map)
    valid_data_loader = object_detector.DataLoader.from_pascal_voc(images_dir=valid_image_dir_path,
                                                                   annotations_dir=valid_label_dir_path,
                                                                   label_map=label_map)
    spec = object_detector.EfficientDetLite0Spec(batch_size=8, model_dir=model_output_dir_path,
                                                 tflite_max_detections=max_detections)

    model = object_detector.create(train_data=train_data_loader,
                                   validation_data=valid_data_loader,
                                   model_spec=spec,
                                   do_train=False,
                                   train_whole_model=True)
    checkpoint = tf.train.Checkpoint(model.model)
    checkpoint.restore(ckpt_path).expect_partial()
    model.export(export_dir=model_output_dir_path,
                 tflite_filename=f'efficientdet-lite0.tflite')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export')
    parser.add_argument('--train_image_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/train')
    parser.add_argument('--train_label_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/train')
    parser.add_argument('--valid_image_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--valid_label_dir_path', type=str, default='~/.vaik-mnist-detection-dataset/valid')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-mnist-detection-dataset/classes.txt')
    parser.add_argument('--ckpt_path', type=str, default='~/output_model/ckpt-20')
    parser.add_argument('--model_output_dir_path', type=str, default='~/export_model')
    parser.add_argument('--max_detections', type=int, default=100)
    args = parser.parse_args()

    args.train_image_dir_path = os.path.expanduser(args.train_image_dir_path)
    args.train_label_dir_path = os.path.expanduser(args.train_label_dir_path)
    args.valid_image_dir_path = os.path.expanduser(args.valid_image_dir_path)
    args.valid_label_dir_path = os.path.expanduser(args.valid_label_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.ckpt_path = os.path.expanduser(args.ckpt_path)
    args.model_output_dir_path = os.path.expanduser(args.model_output_dir_path)

    main(**args.__dict__)
