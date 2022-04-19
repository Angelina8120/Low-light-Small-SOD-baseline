## Catching Small Persons/Vehicles in Low Light: A New SOD Benchmark
by Xin Xu, Shiqin Wang, Zheng Wang, Chia-Wen Lin, Meng Wang

- Since our paper is still under review, we will release our dataset and code once the paper is accepted. Thanks for your understanding.

## Low lIght Salient Person/vehicle (LISP) dataset
- Recent years have witnessed rapid progress if Salient Object Detection (SOD). However, relatively few efforts have been dedicated to modeling salient object detection in low-light scenes with small persons/vehicles. Furthmore, realistic applications of salient person/vehicle detection at long distances in low-light environments commonly exist in nighttime surveillance and nighttime autonomous driving. In particular, for autonomous driving at night, detecting people/vehicles with high reliability is paramount for safety. To fill the gap, we elaborately collect a new Low lIght Salient Person/vehicle (LISP) dataset, which consists of 1,000 high-resolution images containing low-light small persons/vehicles, and covers diverse challenging cases (e.g., low-light, non-uniform illumination environment, and small objects).

- LISP dataset are not openly available due to human data and are available upon resonable request for academic use and within the limitations of the provided informer consent upon acceptance. By downloading the dataset, you guarantee that you will use this dataset for academic work only. 

- Comparison of LISP with existing SOD datasets 

![comparison](./fig/comparison.png)

- Representative images and corresponding ground-truth masks in the LISP dataset

![representative](./fig/representative.png)

## Introduction
![framework](./fig/framework.png) Architecture of Edge and Illumination-Guided Network (EIGNet). It consists of an encoder and three decoders, i.e., a Shared Encoder for feature extraction, an Illumination-Guided Network (IGN), a Saliency Decoder, and an Edge-Guided Network (EGN). The latter three decoders generate the Illumination Map, Saliency Map, and Edge Map, respectively. The decoder progressively integrates IGN and EGN to guide the Saliency Decoder to generate saliency maps in a supervised manner. Among them, IGN applies the Illumination Guidance Layer (IGL) to augment salient features with illumination features.

## Prerequisites
- [Python 3.5](https://www.python.org/)
- [Pytorch 1.3.1](http://pytorch.org/)
- [OpenCV 4.4.0.42](https://opencv.org/)
- [Numpy 1.16.2](https://numpy.org/)
- [TensorboardX 2.1](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)


## Clone repository
```shell
git clone https://github.com/Angelina8120/Low-light-Small-SOD-baseline.git
cd Low-light-Small-SOD-baseline/
```

## Download dataset
Download the following datasets and unzip them into `data` folder

Most of the existing RGB datasets contain multi-scale salient object images, but a large-scale dataset particularly designed for addressing small SOD problems is still missing. To address this issue, we propose a Zoom Out Salient Object (ZOSO) strategy to generate a synthetic normal-light small object (small DUTS-TR) dataset for training.
- small DUTS-TR 

We conduct experiments on our proposed LISP dataset and five widely used datasets, ECSSD, PASCAL-S, DUTS, DUT-OMRON, and SOD.
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [SOD](https://elderlab.yorku.ca/SOD/SOD.zip)
- DUTS: [Google](https://drive.google.com/file/d/1ivK2BCJN8B9UkX_Psf4WF5UcCyxFsTi3/view?usp=sharing) | [Baidu 提取码:ak5t](https://pan.baidu.com/s/1l5UIQYVNRDAX9qg-T09R-g)

Morever, to validate the importance of LISP and the effectiveness of EIGNet for real low-light scenes, we randomly select 500 images from LISP as the training set (LISP-Train), and the other 500 images as the testing set (LISP-Test).

## Training & Evaluation
- Awaiting soon...

## Testing & Evaluate
- Awaiting soon...

## Saliency maps & Trained model
- saliency maps: [Google](https://drive.google.com/file/d/1TXnacKmau7EKoO0Q-_M4Gvnu3BpqCzuJ/view?usp=sharing) | [Baidu 提取码:zywz](https://pan.baidu.com/s/1FWQDvzfcHairkLyqTNdYMQ)

- Quantitative comparisons 

![performace](./fig/table.png)

- Qualitative comparisons 

![sample](./fig/visual.png)

- If you have any questions about the LISP dataset and EIGNet, please contact wangshiqin@wust.edu.cn


