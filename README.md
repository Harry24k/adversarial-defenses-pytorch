# Adversairal-Defenses-Pytorch
Benchmarks list of Adversarial defenses on PyTorch.

[TOC]

## Usage

### Dependencies

- torchattacks 2.0 [[Repo](https://github.com/Harry24k/adversairal-attacks-pytorch)]
- torch 1.5.0
- torchvision 0.5.0
- python 3.6



## Experiment

### Settings
* **Environment**:
	
	* Single NVIDIA Corporation GV100 [TITAN V]
* **Model Architecture**:

  * WRN-28-10 (Dropout=0.3) [[Paper](https://arxiv.org/abs/1605.07146)] [[Code](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py)] is used as a base network.
  * PreActRensNet18 [[Paper](https://arxiv.org/abs/1603.05027)] [[Code](https://github.com/kuangliu/pytorch-cifar)] is used as an additional network.
* **Data Preprocessing**

	* **Batch_size**: 128.
	* **Normalize**: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]. It is included inside WRN due to _torchattacks_.
	* **Augmentation**: _Random Crop, Random Horizontal Flip_ for only _training set_.
* **Training Recipes**: The base setting is same as in [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/)
    * **Stepwise** :
        * **Epoch**: 200.
        * **Optimizer**: SGD with _learning rate=0.1, momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate Decay**: x0.2 when the epoch reaches 60, 120 and 160.
    * **Cyclic**:
        * **Epoch**: 30.
        * **Optimizer**: SGD with _learning rate=0.1, momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate Decay**: _max learning rate=0.2_
* **Google Drive**: For easy reproduction, the state dicts of _pretrained models_ and _adversarial image datasets_ are provided through [Google Drive](https://drive.google.com/drive/folders/1aGcq-mTz0jm6MLS2m3RS6aMwpjRdOrBp?usp=sharing).



## Defenses
### Implementation

Because each defense is totally different, the code is divided into three parts as shown in the below table. For each part, if the defense uses a **different method compared to the base's**, it is checked with :heavy_check_mark:, otherwise it is marked with :x:.

| No.  | Defense  | Description                                                  | Loader | Model |       Train        |
| :--: | -------- | ------------------------------------------------------------ | :----: | :---: | :----------------: |
| #00  | Base     | Clean Training                                               |   -    |   -   |         -          |
| #01  | Adv_FGSM | Adverarial Training with FGSM ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |  :x:   |  :x:  | :heavy_check_mark: |
| #02  | Adv_PGD  | Adverarial Training with PGD ([Madry et al., 2017](https://arxiv.org/abs/1706.06083)) |  :x:   |  :x:  | :heavy_check_mark: |
| #03  | Free     | Adversarial Training for Free! ([Shafahi et al., 2019](https://arxiv.org/abs/1904.12843)) |  :x:   |  :x:  | :heavy_check_mark: |
| #04  | Fast     | Fast is better than Free ([Wong et al., 2020](https://arxiv.org/abs/2001.03994)) |  :x:   |  :x:  | :heavy_check_mark: |



### Reported Accuracy

Reported accuracy of each defense under _epsilon=8/255_ :

| Defense  | Architecture   | Natural | FGSM | PGD7 | PGD20 | PGD50 | Remarks            |
| -------- | -------------- | ------: | ---: | ---: | ----: | ----: | ------------------ |
| Adv_FGSM | ResNet32       |    87.4 | 90.9 |  0.0 |   0.0 |     - |                    |
| Adv_FGSM | WRN-32-10      |    90.3 | 95.1 |  0.0 |   0.0 |     - |                    |
| Adv_PGD  | ResNet32       |    79.4 | 51.7 | 47.1 |  43.7 |     - | _7 steps training_ |
| Adv_PGD  | WRN-32-10      |    87.3 | 56.1 | 50.0 |  45.8 |     - | _7 steps training_ |
| Free     | WRN-32-10      |    86.0 |    - |    - |  46.3 |     - | _PGD restart=10_   |
| Fast     | PreActResNet18 |    86.1 |    - |    - |     - |  46.1 | _PGD restart=10_   |



## Results

### CIFAR10

Top-1 accuracy on CIFAR-10.

Each column means as follows :

* **FGSM**: **White Box** Attack with FSGM(_epslion_=_8/255)_.
* **PGD**: **White Box** Attack with PGD_(stepsize, epsilon, steps=2/255, 8/255, 50)_.
* **RPGD**: **White Box** Attack with RPGD_(stepsize, epsilon, steps=2/255, 8/255, 50) with 10 Random Restarts_.
* **BB1**: **Black Box** Attack with PGD_(stepsize, epsilon, steps=2/255, 8/255, 50)_ using **WRN-28-20** as a Holdout model.
* **BB2**: **Black Box** Attack with PGD_(stepsize, epsilon, steps=2/255, 8/255, 50)_ using **WRN-40-10** as a Holdout model.

Additionally, Holdout models show following accuracy.
* **WRN-28-20** : Clean(96.33%) / FGSM(47.08%) / PGD(0.01%) / RPGD(0.00%)
* **WRN-40-10** : Clean(96.17%) / FGSM(56.18%) / PGD(0.00%) / RPGD(0.00%)



#### WRN-28-10

| Defense  | Decay    | Params                               |    Clean |     FGSM |      PGD |     RPGD |      BB1 |      BB2 | Time(_h_) |
| -------- | -------- | ------------------------------------ | -------: | -------: | -------: | -------: | -------: | -------: | --------: |
| Base     | Stepwise | -                                    | **96.3** |     47.3 |      0.0 |      0.0 |      2.7 |      5.8 |     10.3h |
|          | Cyclic   | -                                    |     94.5 |     11.8 |      0.0 |      0.0 |      7.2 |      9.7 |      1.4h |
| Adv_FGSM | Stepwise | _eps=8/255_                          |     84.1 | **99.2** |      0.1 |      0.0 |     84.2 |     84.7 |     19.4h |
|          |          | Ealry Stopping (_epoch=112/200_)     |     83.3 |     51.8 |     41.7 |     41.3 |     82.2 |     82.2 |         - |
|          | Cyclic   | _eps=8/255_                          |     61.6 |     98.5 |      0.0 |      0.0 |     68.0 |     68.3 |      2.4h |
|          |          | Ealry Stopping (_epoch=20/30_)       |     72.1 |     45.0 |     39.5 |     39.1 |     70.2 |     70.3 |         - |
| Adv_PGD  | Stepwise | _eps=8/255, stepsize=2/255, steps=7_ |     87.0 |     55.5 |     42.2 |     41.1 |     86.0 |     86.2 |     67.5h |
|          | Cyclic   | _eps=8/255, stepsize=2/255, steps=7_ |     85.4 |     59.1 | **51.7** | **51.3** |     84.4 |     84.4 |     11.1h |
| Free     | Stepwise | _eps=8/255, m=8_                     |     85.2 |     54.6 |     45.0 |     44.7 |     84.3 |     84.2 |      9.3h |
| Fast     | Stepwise | _eps=8/255, stepsize=10/255_         |     88.8 |     96.2 |      3.0 |      1.9 | **88.6** | **88.6** |     18.0h |
|          |          | Ealry Stopping (_epoch=129/200_)     |     88.1 |     56.5 |     40.5 |     40.1 |     87.0 |     87.2 |         - |
|          | Cyclic   | _eps=8/255, stepsize=10/255_         |     78.9 |     97.2 |      0.0 |      0.0 |     78.6 |     78.8 |      2.4h |
|          |          | Ealry Stopping (_epoch=25/30_)       |     79.8 |     47.8 |     39.2 |     38.9 |     78.2 |     78.3 |         - |



#### PreActResNet18

