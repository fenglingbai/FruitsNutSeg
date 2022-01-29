# FruitsNutsSeg
 
# 目标：

<font color=#999AAA >学习detectron2数据集的注册以及基本的训练推理
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 一.工程文件下载与数据集准备：
整体的工程文件下载地址：
https://github.com/fenglingbai/FruitsNutSeg
在项目中如图所示(output/model_final_f10217.pkl需要在官网下载)：
![在这里插入图片描述](https://img-blog.csdnimg.cn/b799ae313de34f719073a2304c857895.png?)

水果坚果的实例分割网络这里采用mask_rcnn_R_50_FPN_3x作为例子，网络结构参数在工程文件中已经设置好，如果后续需要更改可以自己进行。
网络预训练权重在官网的
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
进行相应权重的下载(标红点处)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b224dd4e50ae475cb201de9e5ee66c79.png?)
放在output文件夹中。
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 二.注册数据集：
在fruitsnuts_data.py文件中，进行水果坚果数据集的注册。代码如下所示
```python
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
register_coco_instances("fruits_nuts", {'thing_classes':['date', 'fig', 'hazelnut'],
                                        'thing_dataset_id_to_contiguous_id':{1: 0, 2: 1, 3: 2}}, "./data/trainval.json", "./data/images")
```
其中函数register_coco_instances是detectron2的一个接口，用来专门注册coco形式的数据集，其基本形式如下：

```python
register_coco_instances(name, metadata, json_file, image_root)
```
其中name是自己的数据集名称，json_file是标注文件的地址，image_root是图片数据的地址，metadata是数据集的基本信息，如果缺省则用空字典输入，否则可以用键值对的形式录入。
注册完毕后，可调用接口进行检查，注释为对应的数据集数据：
```python
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
# Metadata(evaluator_type='coco', image_root='./data/images',
# json_file='./data/trainval.json', 
# name='fruits_nuts', thing_classes=['date', 'fig', 'hazelnut'], 
# thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2})
```
在其他文件进行注册时，可以直接使用

```python
import fruitsnuts_data
```
进行注册
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 训练：
训练代码详见train.py：
```python
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import fruitsnuts_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os
setup_logger()

if __name__ == "__main__":
    cfg = get_cfg()
    # cfg.merge_from_file(
    #     "../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.MODEL.WEIGHTS = "./output/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = (300)  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print('OK')
```
注意需要修改的地方主要有以下几点：
1.cfg.merge_from_file：网络参数设置的路径
2.cfg.MODEL.WEIGHTS：网络初始化权重的路径
运行后直接可以进行网络训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/d91b408a0f3942dbb8771e4150dfe98d.png?)
训练后的结果为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a1b7a878921742cca0fbc32214e30a94.png?)

# 推理：
预测推理代码详见predict.py：
```python
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import fruitsnuts_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")

if __name__ == "__main__":
    cfg = get_cfg()
    # cfg.merge_from_file(
    #     "../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ("fruits_nuts", )
    predictor = DefaultPredictor(cfg)

    data_f = './data/images/3.jpg'
    im = cv2.imread(data_f)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)

```
其中data_f是需要预测推理的图片地址，运行程序后结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/bf0c2a0d19034fd7adf6d575a137c12a.png?)

# 三.其他可能的问题：
## 1.预测的图片没有类别标注
这是由于在数据注册阶段没有进行对应的数据输入，
## 2.出现警告UserWarning: This overload of nonzero is deprecated： nonzero
解决方法：在对应位置nonzero()中改为：
nonzero(as_tuple=False)
## 3.出现警告Skip loading parameter
这是由于初始的网络权重是适应81类分类任务的，而当前的分类任务为4，因此会有一些网络参数进行省略，这里忽略即可。
