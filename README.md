# Adversarial-Defenses-PyTorch
_Adversarial-Defenses-PyTorch_ is a set of implementations of adversarial defense methods. The goal of this package is to make a _fair comparison_ of adversarial defenses. Although there's a lot of adversarial defense methods, a fair comparison between methods is still difficult. There are two reasons: (1) _different coding style_, and (2) _different training setting_. To resolve this issue, in this repo, we provide several defense methods and their robustness with the same setting.



## Usage

### :clipboard: Dependencies

- python==3.6
- torch==1.4.0
- torchvision==0.5.0
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)==2.11
  - `pip install torchattacks`
  - `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`
- [torchhk](https://github.com/Harry24k/pytorch-custom-utils)==0.85
  - `pip install torchhk`
  - `git clone https://github.com/Harry24k/pytorch-custom-utils`



## Citation

If you use this package, please cite the following BibTex:

```
@article{kim2020torchattacks,
  title={Torchattacks: A Pytorch Repository for Adversarial Attacks},
  author={Kim, Hoki},
  journal={arXiv preprint arXiv:2010.01950},
  year={2020}
}
```



## List of defenses
**The code** is divided into three parts, that are data **loader**, **model** structure, and **trainer**. For each part, if the defense uses a different method compared to the base's, it is checked with :heavy_check_mark:, otherwise it is marked with :x:.

| Defense       | Description                                                  | Loader | Model |      Trainer       |
| ------------- | ------------------------------------------------------------ | :----: | :---: | :----------------: |
| **Base**      | **Standard Training**                                        |   -    |   -   |         -          |
|               | **Single-step Adversarial Training**                         |        |       |                    |
| **FGSMAdv**   | Adversarial Training with FGSM ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |  :x:   |  :x:  | :heavy_check_mark: |
| **Free**      | Adversarial Training for Free! ([Shafahi et al., 2019](https://arxiv.org/abs/1904.12843)) |  :x:   |  :x:  | :heavy_check_mark: |
| **Fast**      | Fast is better than Free ([Wong et al., 2020](https://arxiv.org/abs/2001.03994)) |  :x:   |  :x:  | :heavy_check_mark: |
| **GradAlign** | Understanding and Improving Fast Adversarial Training ([Andriushchenko et al., 2020]()) |  :x:   |  :x:  | :heavy_check_mark: |
|               | **Multi-step Adversarial Training**                          |        |       |                    |
| **PGDAdv**    | Adversarial Training with PGD ([Madry et al., 2017](https://arxiv.org/abs/1706.06083)) |  :x:   |  :x:  | :heavy_check_mark: |
| **TRADES**    | Theoretically Principled Trade-off between Robustness and Accuracy ([Zhang et al., 2019](https://arxiv.org/abs/1901.08573)) |  :x:   |  :x:  | :heavy_check_mark: |
| **MART**      | Improving Adversarial Robustness Requires Revisiting Misclassified Examples ([Wang et al., 2020](https://openreview.net/forum?id=rklOg6EFwS)) |  :x:   |  :x:  | :heavy_check_mark: |



## Validation with Reported Accuracy

| Defense  | Architecture   | Natural | FGSM | PGD7 | PGD20 | PGD50 | Remarks            |
| -------- | -------------- | ------: | ---: | ---: | ----: | ----: | ------------------ |
| Adv_FGSM | ResNet32       |    87.4 | 90.9 |  0.0 |   0.0 |     - |                    |
| Adv_FGSM | WRN-32-10      |    90.3 | 95.1 |  0.0 |   0.0 |     - |                    |
| Adv_PGD  | ResNet32       |    79.4 | 51.7 | 47.1 |  43.7 |     - | _7 steps training_ |
| Adv_PGD  | WRN-32-10      |    87.3 | 56.1 | 50.0 |  45.8 |     - | _7 steps training_ |
| Free     | WRN-32-10      |    86.0 |    - |    - |  46.3 |     - | _PGD restart=10_   |
| Fast     | PreActResNet18 |    86.1 |    - |    - |     - |  46.1 | _PGD restart=10_   |



## Settings

* **Environment**:
	* Single NVIDIA Corporation GV100 [TITAN V]
* **Model Architecture**:
    * **Pre-Act-ResNet-18** [[Paper](https://arxiv.org/abs/1603.05027)] [[Code](https://github.com/kuangliu/pytorch-cifar)]
        * #Params: 11,171,146
    * **Wide-ResNet-28-10** [[Paper](https://arxiv.org/abs/1605.07146)] [[Code](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py)]
      * #Params: 36,479,194
* **Data Preprocessing**
  * **Normalize**: _mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]_. Please refer to [here](/defenses/model.py).
  * **Augmentation**: _Random Crop, Random Horizontal Flip_ for only _training set_.
* **Training Recipes**:
    * **Stepwise**:
        * **Epoch**: 100.
        * **Optimizer**: _SGD_ with _momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate**: _Initial learning rate=0.1_ and _decay x0.1 at 50 and 75 epoch_.
    * **Cyclic**:
        * **Epoch**: 30.
        * **Optimizer**: _SGD_ with _momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate Decay**: _Initial learning rate=0.0_ and _maximum learning rate=0.3_



## Benchmarks

### CIFAR10





#### Pre-Act-ResNet-18