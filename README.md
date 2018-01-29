# MobileNet V2 using Keras

Author: Shengdong Zhang
Email: sza75@sfu.ca

This is the beta implementation of Mobilenet V2(https://128.84.21.199/pdf/1801.04381.pdf) using Keras.

Because there are still some contradiction in the model description part in the paper, this script is implemented based on the best understanding of the script author. Updates will be made as soon as it is ready or the paper is updated.

MobileNet v2 is the next version of MobileNet v1 with big improvement. Instead of directly using depthwise convolution + 1x1 convolution structure, it implements inverted residual block structure by first expanding input data into a larger dimension and then applying 3x3 depthwise convolution plus 1x1 convolution bottlenet structure to decrease dimension. Based on the experiments in the paper and my personal experience, this structure does help gradients pass through the deep network which leverage the gradient vanishing problem.

Currently modification needed if you want to use the script for images with small sizes like CIFAR10 or MNIST. ImageNet pretrained weights will be released as soon as it is available.

The following table describes the size and accuracy of different light-weight networks on size 224 x 224:
-----------------------------------------------------------------------------
Network                  |   Top 1 acc   |  Multiply-Adds (M) |  Params (M) |
-----------------------------------------------------------------------------
|   MobileNetV1          |    70.6 %     |        575         |     4.2     |
|   ShuffleNet (1.5)     |    69.0 %     |        292         |     2.9     |
|   ShuffleNet (x2)      |    70.9 %     |        524         |     4.4     |
|   NasNet-A             |    74.0 %     |        564         |     5.3     |
|   MobileNetV2          |    71.7 %     |        300         |     3.4     |
-----------------------------------------------------------------------------

# Reference
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation]
(https://arxiv.org/pdf/1801.04381.pdf))
