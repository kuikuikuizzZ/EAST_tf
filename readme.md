# EAST: An Efficient and Accurate Scene Text Detector

这里EAST的代码是参考了https://github.com/argman/EAST的代码实现，也做了以下的改进

- 对数据读取的部分进行了优化，主要体现在icdar.py的generate_rbox函数上，对预处理步骤向量化处理，提高了速度，cpu处理不再是性能瓶颈
- 重写了专门的generator支持多线程，继承keras.utils.Sequence，可以保证性能和简洁

- 去掉data_utils.py 等数据多线程处理的文件，更新了相关的接口，直接使用keras底层代码，代码更加简单易懂
- 对部分艰涩代码提供注释

### 使用说明

- 数据准备

  - 需要把数据整理成如training_samples的结构，bounding box 的信息放在txt中，然后分别整理出images文件夹路径（放在一起也可以）---> path/to/images_dir, path/to/txt_dir

  - 注意txt文件的保存格式是如下：

    **x1,y1,x2,y2,x3,y3,x4,y4, label ** 

    377,117,463,117,465,130,378,130,Genaxis Theatre
    493,115,519,115,519,131,493,131,[06]
    374,155,409,155,409,170,374,170,###

    需要注意的是这里的坐标是按照顺时间的顺序排列的左上为p1,左下为p4

- 训练过程

  可以按照上面给出的data位置，和你的硬件设置改变以下的参数，具体可以参考multigpu-train.py文件

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size=14 
						 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ 
    					 --workers=0  --images_dir=/data/ocr/icdar2015/ 
        				 --txt_dir=/data/ocr/icdar2015/ 
                         --learning_rate=0.0001 --use_multiprocessing=True
					     --pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

- 结果验证

  可以使用test/test_EAST.ipynb尝试去或者模型的效果，已经实现了EAST_Pridictor，作为一个预测api。具体可以参考文件里面的信息

- 下面原来repository的readme，一些安装信息和数据集的信息都可以在下面获取到，上面就不赘述了。

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:
+ Online demo
	+ http://east.zxytim.com/
	+ Result example: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
	+ CAVEAT: There's only one cpu core on the demo server. Simultaneous access will degrade response time.
+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the paper's author.
+ The pre-trained model provided achieves **80.83** F1-score on ICDAR 2015
	Incidental Scene Text Detection Challenge using only training images from ICDAR 2015 and 2013.
  see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_samples&task=1&m=29855&gtv=1) for the detailed results.
+ Differences from original paper
	+ Use ResNet-50 rather than PVANET
	+ Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
	+ Use linear learning rate decay rather than staged learning rate decay
+ Speed on 720p (resolution of 1280x720) images:
	+ Now
		+ Graphic card: GTX 1080 Ti
		+ Network fprop: **~50 ms**
		+ NMS (C++): **~6ms**
		+ Overall: **~16 fps**
	+ Then
		+ Graphic card: K40
		+ Network fprop: ~150 ms
		+ NMS (python): ~300ms
		+ Overall: ~2 fps

Thanks for the author's ([@zxytim](https://github.com/zxytim)) help!
Please cite his [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Demo](#demo)
3. [Test](#train)
4. [Train](#test)
5. [Examples](#examples)

### Installation
1. Any version of tensorflow version > 1.0 should be ok.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Demo
If you've downloaded the pre-trained model, you can setup a demo server by
```
python3 run_demo_server.py --checkpoint-path /tmp/east_icdar2015_resnet_v1_50_rbox/
```
Then open http://localhost:8769 for the web demo. Notice that the URL will change after you submitted an image.
Something like `?r=49647854-7ac2-11e7-8bb7-80000210fe80` appends and that makes the URL persistent.
As long as you are not deleting data in `static/results`, you can share your results to your friends using
the same URL.

URL for example below: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
![web-demo](demo_images/web-demo.png)


### Test
run
```
python eval.py --test_data_path=/tmp/images/ --gpu_list=0 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=/tmp/
```

a text file will be then written to the output path.


### Examples
Here are some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)

### Troubleshooting
+ How to compile lanms on Windows ?
  + See https://github.com/argman/EAST/issues/120

Please let me know if you encounter any issues(my email boostczc@gmail dot com).
