# Accurate identification and measurement of the precipitate area by two-stage deep neural networks in novel chromium-based alloys

## About The Project

The performance of advanced materials for extreme environments is underpinned by their microstruc- ture, such as the size and distribution of nano- to micro-sized reinforcing phase(s). Chromium-based superalloys are a recently proposed alternative to conventional face-centred-cubic superalloys for high-temperature applications, e.g., Concentrated Solar Power. Their development requires the de- termination of precipitate volume fraction and size distribution using Electron Microscopy (EM), as these properties are crucial for the thermal stability and mechanical properties of chromium superal- loys. Traditional approaches to EM image processing utilise filtering with a fixed contrast threshold, leading to weak robustness to background noise and poor generalisability to different materials. It also requires an enormous amount of time for manual object measurements. Efficient and accurate object detection and segmentation are therefore highly desired to accelerate the development of novel materials like chromium-based superalloys. To address these bottlenecks, based on YOLOv5 and SegFormer structures, this study proposes an end-to-end, two-stage deep learning scheme, DT- SegNet, to perform object detection and segmentation for EM images. The proposed approach can thus benefit from the training efficiency of Convolutional Neural Networks at the detection stage (i.e., a small number of training images required) and the accuracy of the Vision Transformer at the segmentation stage. Extensive numerical experiments demonstrate that the proposed DT-SegNet significantly outperforms the state-of-the-art segmentation tools offered by Weka and ilastik regard- ing a large number of metrics, including accuracy, precision, recall and F1-score. This model will be a meaningful tool for accelerating alloy development and microstructure examination.

## Getting Started

### Quick Start

You can use the [Inference_Colab.ipynb](./Inference_Colab.ipynb) ([Colab link](https://colab.research.google.com/github/xiazeyu/DT_SegNet/blob/main/Inference_Colab.ipynb)) to perform online inference.

### Hardware requirement

- Operation System: Windows or Linux
- Platform: AutoDL / Google Colab Pro
- GPU: NVIDIA RTX A5000
- Google Drive space: 10GB

### Software requirement

- Programming language: Python (3.8 or higher)
- Package Management: Anaconda (Miniconda recommended)
- Machine Learning Framework: [PyTorch](https://pytorch.org/get-started/locally/) and [PaddlePaddle](https://www.paddlepaddle.org.cn/en/install/quick)

Anaconda environment for labelling on Windows system is in `dtsegnet.yaml`. The environment can be restored by executing `conda create --name dtsegnet --file dtsegnet.yaml` in the console.

Two machine learning frameworks need to be installed following the tutorials on their websites. The necessary environment for training and inferring is stored as a pip requirement file in `1_Detection_Model/requirements.txt` and `3_Segmentation_Model/requirements.txt`.

All the requirements for training and inferring will be installed in the **0_Prepare.ipynb** notebook.

## Dataset

All data for this project are stored in the `Dataset/` folder. All images are numbered for the DT-SegNet pipeline, and the data mapping is stored in [dataset_mapping.csv](./Dataset/dataset_mapping.csv).

The dataset contains the original image, segmentation label and detection label. Detection labels can be used directly for the detection network, but the segmentation label needs to be cropped using codes provided in the notebook before delivering to the segmentation network.

The detection dataset is separated into three sets: `test`, `train`, and `val`.

The segmentation annotation is stored in `Dataset/segmentation_labels/`. The Regions of Interest for the segmentation network with their annotations will be generated before the segmentation stage.


## Implementation

### 0 Prepare

Follow the cells in `0_Prepare.ipynb` to prepare required environments.

### 0 Label the dataset

`0_Labelling_Tools/` contains tools of scripts to label the dataset. The user should follow the following process to label the dataset.

- Execute `conda create --name dtsegnet-label --file 0_Labelling_Tools/dtsegnet-label.yaml` to import the Anaconda environment for labelling.
- Execute `conda activate dtsegnet-label` to activate the labelling environment.
- Download the model for EISeg labelling from [https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_aluminium.zip](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_aluminium.zip).
- Execute `python 0_Labelling_Tools/0_EISeg/exe.py` to start labelling for the segmentation stage.
- Execute `python 0_Labelling_Tools/3_gray2pseudo_color.py <Dataset/label>` to convert the segmentation images to pseudo colour annotation images. **<Dataset/label>** should be replaced by the path of segmentation annotations generated by EISeg.
- Execute `python 0_Labelling_Tools/1_Segmentation_Label_Flood_Fill.py <Dataset/label> <Dataset/Detection_Label>` to generate the detection labels from the segmentation labels. **<Dataset/label>** should be replaced by the path of segmentation annotations generated by EISeg, which contain files like: `1.png`, `1_cutout.png` and `1_pseudo.png`. **<Dataset/Detection_Label>** should be replaced by the output folder for detection labels.
- Execute `python 0_Labelling_Tools/2_labelImg/labelImg.py` to finetune the detection labels.

### 1 Train the model

Follow the cells in `1_Train.ipynb` to train the detection model. The trained detection model will be stored in `<Google Drive>/DT-SegNet/Detection_Model_Output`. it will also automatically generate the dataset for the segmentation network. The generated segmentation dataset will be compressed and stored in `<Google Drive>/DT-SegNet/Segmentation_Dataset.zip`. The trained segmentation model will be stored in `<Google Drive>/DT-SegNet/Segmentation_Model_Output`.

### 2 Inference

Follow the cells in `2_Inference.ipynb` to infer using DT-SegNet. The output from the detection model will be stored in `<Google Drive>/DT-SegNet/Detection_Output`. The output from the segmentation model will be held in `<Google Drive>/DT-SegNet/Segmentation_Output`. The original-size segmentation mask will be stored in `<Google Drive>/DT-SegNet/Output`.

### 3 Validation

Follow the cells in `3_Validation.ipynb` to validate the trained models.

### 4 Analysis

Follow the cells in `4_Analysis.ipynb` to analyse the trained models. To compare software and algorithms, we performed experiments on [Weka trainable segmentation](https://imagej.net/plugins/tws/) and [Ilastik pixel classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification). The notebook uses our output in `Output/`.

### Main Models

Our best-trained models are stored on the [Github release page](https://github.com/xiazeyu/DT_SegNet/releases/).

Machine-friendly download links are also provided for automatic scripts:

```
https://github.com/xiazeyu/DT_SegNet/releases/latest/download/detection.pt
https://github.com/xiazeyu/DT_SegNet/releases/latest/download/segmentation.pdparams
```

### Output

The `Output/` folder holds this project's output images and NumPy metrics, including results from DT-SegNet, Weka and ilastik. Each experiment has an output in two different formats: the `.png` image output and the `.npy` NumPy matrix output. Each file in the folder is named by `<test id>_<software used>_<algorithm>`.

| Software used |   Algorithm       |                            Remark                            |
| :-----------: | :---------------: | :----------------------------------------------------------: |
|  groundtruth  |  groundtruth      |                      Manually annotated                      |
|    ilastik    |      LDA          |                  ilastik LDA (scikit-learn)                  |
|    ilastik    |      RF           |             ilastik Random Forest (scikit-learn)             |
|    ilastik    |      SVC          |             ilastik SVM C-Support (scikit-learn)             |
|     weka      |      FRF          |        Weka hr/irb/fastRandomForest/FastRandomForest         |
|     weka      |      MLP          | weka/classifier/functions/MultilayerPreceptron<br />with trainingTime=100 and validationSetSize=20 |
|    PaddleSeg  |      unet         |             U-Net                                            |
|    PaddleSeg  |      unet_3plus   |             UNet 3+                                          |
|    PaddleSeg  |deeplabv3p_resnet50|             DeepLabV3+ with ResNet 50 Backbone               |
|    PaddleSeg  |      B0           |             SegFormer B0                                     |
|    PaddleSeg  |      B1           |             SegFormer B1                                     |
|   dtsegnet    |   DT-SegNet       |           DT-SegNet, with overlapping ROIs joined            |


## License

MIT License. More information see [LICENSE](./LICENSE)


## Contact

Zeyu Xia - [zeyu.xia@connect.qut.edu.au](mailto:zeyu.xia@connect.qut.edu.au)

Kan Ma - [arnaud.masysu@gmail.com](mailto:arnaud.masysu@gmail.com)

Sibo Cheng - [sibo.cheng@imperial.ac.uk](mailto:sibo.cheng@imperial.ac.uk)

