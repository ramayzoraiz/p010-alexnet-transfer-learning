# p010-alexnet-transfer-learning
AlexNet applied on OxfordIIIT Pet Dataset through transfer learning via pytorch framework. Finetuning and feature extraction were explored and performance was noted.  Effect of synthetic data (through augmentation techniques) was also observed.

# Result
Higher result eas achieved with combination of feature-extraction and then finetuning. **Through transfer learning of AlexNet, $\approx$ 70% was achieved as compared to $\approx$ 32% when trained [AlexNet from scratch] as in [this](https://github.com/ramayzoraiz/p009-alexnet-cnn-model) repository.** Dataset is complex and small due to which training with scratch is infeasible and also time is saved.

> **Note:**
AlexNet model architecture taken from [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997).
>AlexNet was originally introduced in the [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) paper. Pytorch implementation is based instead on the “One weird trick” paper above.


# Dataset and Split
The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200 images for each class. Data has 7349 images of pets and is split into 85:15 train/test data respectively.
# Preprocessing
Images are normalized with dataset mean and std. **Increase of upto 6% accuracy** was observed (can be seen from summarzied table) and it makes sense as AlexNet was also trained on normalized data. 
# Data Augmentation
Data augmentation plays important part in training model especially when data is limited. Dataset size is increased by 6 fold via augmentation techniques in file [pytorch_utils](pytorch_utils.py). Original images are concatenated with central and corners crop. Central, top left, top right, bottom left, bottom left crops are additionally preprocessed by left_right_flip, random_hue, random_brightness, random_saturation, random_contrast; respectively. Moreover, corner crops are randomly left_right flipped using different seeds. **Increase upto 4%** was observed as can seen from later section table.
# AlexNet Architecture
Pytorch open AlexNet model is little modified as there is no LRN (or BatchNorm) layer. Further, in order to make architecture to be able to handle different size inputs, AdaptiveAveragePool2D is used so linear(aka fully connected) layers parameters do not change and others can benefit from transfer learning. 
<pre>
AlexNet(
    (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
</pre>


# AlexNet Result Summary
AlexNet architecture with SGD momentum was trained by transfer learning. Transfer learning can be categorized into finetuning, feature-extraction and combination; here first three files/cases are of finetuning then three of feature-extraction and last two of their combination (feature-extraction and then finetuning). In finetuning, we train the whole model on new data while in feature-extraction, we freeze layers(mostly convolutional) and train only classifier layers or model. \
## Conclusion
Higher result eas achieved with combination of feature-extraction and then finetuning. **Through transfer learning of AlexNet, $\approx$ 70% was achieved as compared to $\approx$ 32% when trained [AlexNet from scratch] as in [this](https://github.com/ramayzoraiz/p009-alexnet-cnn-model) repository.** Dataset is complex and small due to which training with scratch is infeasible.


|  file_name.ipynb | train_acc | val_acc | lr | epoch | train_loss | val_loss | Normalization | Synthetic Data | Transfer Learning Technique |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [pytorch_alexnet_transfer_learning_case1](pytorch_alexnet_transfer_learning_case1.ipynb) | 59.31 | 56.86 | 0.0003,0.00003,0.000003, 0.0000003  | 7,7,7, 4 | 1.3158 | 1.4054 | no  | no  | finetuning |
| [pytorch_alexnet_transfer_learning_case2](pytorch_alexnet_transfer_learning_case2.ipynb) | 64.55 | 64.67 | 0.0002, 0.00002,0.000002,0.0000002  | 7, 7,7,4 | 1.1374 | 1.2099 | yes | no  | finetuning |
| [pytorch_alexnet_transfer_learning_case3](pytorch_alexnet_transfer_learning_case3.ipynb) | 66.96 | 66.85 | 0.0001,0.00001, 0.000001            | 7,7, 4   | 1.0563 | 1.1078 | yes | yes | finetuning |
| [pytorch_alexnet_transfer_learning_case4](pytorch_alexnet_transfer_learning_case4.ipynb) | 63.48 | 61.31 | 0.00025,0.000025, 0.0000025         | 7,7, 4   | 1.2192 | 1.3295 | yes | no | feature-extraction(only last layer trained) |
| [pytorch_alexnet_transfer_learning_case5](pytorch_alexnet_transfer_learning_case5.ipynb) | 60.92 | 60.49 | 0.00015,0.000015,0.0000015          | 7,7,4    | 1.2972 | 1.3767 | yes | no | feature-extraction(classifier further trained) |
| [pytorch_alexnet_transfer_learning_case6](pytorch_alexnet_transfer_learning_case6.ipynb) | 56.59 | 61.22 | 0.0003,0.00003,0.000003             | 9,9,7    | 1.5149 | 1.3916 | yes | no | feature-extraction(classifier scratch trained) |
| [pytorch_alexnet_transfer_learning_case7](pytorch_alexnet_transfer_learning_case7.ipynb) | 66.60 | 65.03 | 0.0002,0.00002,0.000002, 0.0000002  | 7,7,7, 4 | 1.0715 | 1.1468 | yes | no | finetuning on top of above(case6) |
| [**pytorch_alexnet_transfer_learning_case8**](pytorch_alexnet_transfer_learning_case8.ipynb) | **68.65** | **69.75** | 0.0003,0.00003,0.000003, 0.0000003  | 7,7,7, 4 | 0.9903 | 1.0476 | yes | yes | finetuning on top of case6 |









