datasets内为cifar100数据集,下载地址为链接: https://pan.baidu.com/s/1V2ZoIoscT7KKhJ6JEVE-1g 提取码: ikgo 
logs内为tensorboard记录的loss与acc
saved_model内为保存的模型，下载地址为链接: https://pan.baidu.com/s/1YvcOwzQ2zl9p71dg8nIt5A 提取码: wls6 
--来自百度网盘超级会员v5的分享
models.py为搭建的模型，包括导入resnet18模型，查看参数量。
模型参考vision transformer，（vit），仅对模型超参进行调整，如n_head,d_model等
train.py为训练脚本，训练集训练，验证集验证，保存模型，保存训练损失等，训练使用的超参数在最前面的参数解析处。
