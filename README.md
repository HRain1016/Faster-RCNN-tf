# Faster R-CNN TensorFlow版
由Xinlei Chen（xinleic@cs.cmu.edu）实现的Faster R-CNN检测框架的Tensorflow版本。 原作者基于Caffe实现的版本[点击这里](https://github.com/rbgirshick/py-faster-rcnn)。

注意：重新实现框架时进行了一些改动，有关详细信息，请参阅技术报告[An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf)。 如果想重现原论文的结果，请使用[官方代码](https://github.com/ShaoqingRen/faster_rcnn)或[半官方代码](https://github.com/rbgirshick/py-faster-rcnn)。 有关Faster R-CNN架构的详细信息，请参阅原论文[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)。

### 检测表现
当前代码支持VGG16、Resnet V1和Mobilenet V1。唯一的数据增强技术是在原始Faster RCNN之后的训练期间左右翻转。

使用VGG16（conv5_3）：
* 在VOC 2007 训练验证集训练并在VOC 2007测试集测试，70.8。
* 在VOC 2007+2012 训练验证集训练并在VOC 2007测试集测试，75.7。
* 在COCO 2014 trainval35k上训练并在minival上测试（迭代次数：900k / 1190k），30.2。

使用Resnet101（最后一个conv4）：
* 在VOC 2007 训练验证集训练并在VOC 2007测试集测试，75.7。
* 在VOC 2007+2012 训练验证集训练并在VOC 2007测试集测试，79.8。
* 在COCO 2014 trainval35k上训练并在minival上测试（迭代次数：900k / 1190k），35.4。

更多结果：
* 在COCO 2014 trainval35k上训练Mobilenet（1.0,224）并在minival上测试（900k / 1190k），21.8。
* 在COCO 2014 trainval35k上训练Resnet50并在minival上测试（900k / 1190k），32.4。
* 在COCO 2014 trainval35k上训练Resnet152并在minival上测试（900k / 1190k），36.1。

#### 注意：
由于保留了小的候选区域（<16像素宽度/高度），检测器特别适合检测小型物体。

### 附加功能
* __支持train-and-validation__。在训练期间，将实时测试验证数据以监控训练过程并检查潜在的过拟合。理想情况下，训练和验证应该是分开的，每次加载模型以进行验证测试。但是我已经以联合的方式实现它以节省时间和GPU内存。

* __支持恢复培训__。尝试在快照时存储尽可能多的信息，目的是正确地从最新快照恢复训练。

* __支持可视化__。当前的实现将在训练期间总结ground truth boxes，统计损失、激活函数和变量，并将其转储到单独的文件夹以进行tensorboard可视化。计算图也会保存以供调试。

### 环境需求
cython，opencv-python，easydict 1.6，tensorflow1.5
### 安装
1. 下载仓库
```
git clone https://github.com/endernewton/tf-faster-rcnn.git
```
2. 更新-arch匹配GPU
```
cd tf-faster-rcnn/lib`
# Change the GPU architecture (-arch) if necessary
vim setup.py
```
 GPU model  | Architecture
 ---- | ----- 
TitanX (Maxwell/Pascal)  | sm_52
GTX 960M | sm_50
GTX 1080 (Ti) | sm_61
Grid K520 (AWS g2.2xlarge) | sm_30
Tesla K80 (AWS p2.xlarge) | sm_37

3. 构建 Cython modules
```
make clean
make
cd ..
```
4. 安装 Python COCO API，使用该 API 连接 COCO 数据库
```
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```
### 设置数据
请按照[py-faster-rcnn的说明](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)设置VOC和COCO数据集（COCO的一部分已完成）。 这些步骤涉及下载数据并可选地在数据文件夹中创建软链接。

### 使用预训练模型演示
1. 下载预训练模型
```
# Resnet101 for voc pre-trained on 07+12 set
./data/scripts/fetch_faster_rcnn_models.sh
```
2. 创建文件夹和软连接使用预训练模型
```
NET=res101
TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
mkdir -p output/${NET}/${TRAIN_IMDB}
cd output/${NET}/${TRAIN_IMDB}
ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
cd ../../..
```
3. 在自定义图片上测试
```
# at repository root
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```
4. Test with pre-trained Resnet101 models
```
GPU_ID=0
./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
```
注意: 如果不能得到报告中的结果(79.8 ), 可能是因为 NMS 函数编译有问题。

### 训练自己的模型
1. 下载预先训练的模型和权重。 目前的代码支持 VGG16 和 Resnet V1模型。 预训练模型由slim提供，[点击这里](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)获取预训练模型并将它们设置在data / imagenet_weights文件夹中。 例如对于VGG16，可以设置如下：
```
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
```
对于Resnet101, 可以设置如下:
```
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```
2. 训练
```
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
# Examples:
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/train_faster_rcnn.sh 1 coco res101
```
注意: 使用预训练模型训练前删除软连接。

3. 使用Tensorboard可视化
```
tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
```
4. 测试和评估
```
./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# GPU_ID is the GPU you want to test on
# NET in {vgg16, res50, res101, res152} is the network arch to use
# DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
# Examples:
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/test_faster_rcnn.sh 1 coco res101
```
5. 使用 tools/reval.sh 重新评估
* 默认训练网络存储在:
```
output/[NET]/[DATASET]/default/
```
* 测试输出存储在:
```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```
* 训练和验证的Tensorboard信息存储在:
```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```
