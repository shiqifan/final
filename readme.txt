目录：
faster_rcnn_modify内两个脚本替换掉源代码nets中对应的脚本，新增从mask-rcnn导入coco预训练resnet50主干网络的脚本。
loss_and_map，有训练好的权重pth文件，loss曲线，loss和map的tensorboard

源代码：
train中，model_path都设置为空
源代码选backbone时，在 pretrained      = True，backbone = 'resnet50'，是使用imageNet的预训练权重，backbone = 'resnet50coco'从mask-rcnn导入coco预训练resnet50主干网络。
pretrained      = False，backbone = 'resnet50'，使用随机初始化的权重训练
