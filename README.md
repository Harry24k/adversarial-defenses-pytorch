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



### Train

**1. Modify `settings.py` to set arguments for training.**

   ```python
args.gpu = 0
args.name = "FGSMAdv_CIFAR10_PRN18_Cyclic"
args.save_path = "./_checkpoint/"
args.loader = "Base"
args.model = "PRN18"
args.trainer = "FGSMAdv"
...
   ```

Options for loader, model, and trainer are in [loader.py](/defenses/loader.py), [model.py](/defenses/model.py), and [trainer.py](/defenses/trainer.py).



**2. Run train.py**

```bash
python train.py
```



Then, training records will be printed every epoch.

* Clean: Average standard accuracy.
* FGSM: Average robust accuracy against FGSM.
* PGD: Average robust accuracy against PGD.
* GN: Average robust accuracy against Guassian Noise with a standard deviation 0.1.

Each record will be evaluated on the first training batch (Tr) and the first test batch (Te).

At the end of training, it shows a summary of the training records. In addition, the weights of model will be saved to [save_path+ name + ".pth"] (e.g., ./checkpoint/sample.pth) and the records will also be saved to [save_path+ name + ".csv"] (e.g., ./checkpoint/sample.csv).



### Evaluation

#### Standard Accuracy
```bash
python eval.py --model-path "./_checkpoint/model.pth" --model "PRN18" --data "CIFAR10" --gpu 0 --method "Standard"
```

#### FGSM

```bash
python eval.py --model-path "./_checkpoint/model.pth" --model "PRN18" --data "CIFAR10" --data-path "FGSM.pt" --gpu 0 --method "FGSM" --eps 8
```

#### PGD50

```bash
python eval.py --model-path "./_checkpoint/model.pth" --model "PRN18" --data "CIFAR10" --data-path "PGD.pt" --gpu 0 --method "PGD" --eps 8 --alpha 2 --steps 50 --restart 1
```

These commands will show you the robust accuracy against FGSM or PGD for the test dataset and save the adversarial images to the data-path.



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
    * **Pre-Act-ResNet-18 (PRN18)**  [[Paper](https://arxiv.org/abs/1603.05027)] [[Code](https://github.com/kuangliu/pytorch-cifar)]
      
        * #Params: 11,171,146
    * **Wide-ResNet-28-10 (WRN28)** [[Paper](https://arxiv.org/abs/1605.07146)] [[Code](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py)]
      
      * #Params: 36,479,194
      
      


* **Data Preprocessing**
  * **Normalize**: _mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]_. Please refer to [here](/defenses/model.py).
  * **Augmentation**: _Random Crop, Random Horizontal Flip_ for only _training set_.
  
  


* **Training Recipes**:
    * **Stepwise**:
        * **Epoch**: 100.
        * **Optimizer**: _SGD_ with _momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate**: _Initial learning rate=0.01_ and _decay x0.1 at 50 and 75 epoch_.
    * **Cyclic**:
        * **Epoch**: 30.
        * **Optimizer**: _SGD_ with _momentum=0.9, weight_decay=5e-4_.
        * **Learning Rate Decay**: _Initial learning rate=0.0_ and _maximum learning rate=0.3_



## Robustness

Following "Overfitting in adversarially robust deep learning ([Rice et al., 2020](https://arxiv.org/abs/2002.11569))", we summarize robustness at the epoch of the best PGD accuracy on the first test batch for stepwise learning rate schedule.

Checkpoints for each model are provided through the link of each model name.

### CIFAR10

#### Cyclic

| PRN18         | Standard |     FGSM |      PGD | Time (m) |
| ------------- | -------: | -------: | -------: | -------: |
| **Base**      |     93.3 |     14.7 |      0.0 |       13 |
| _Single-step_ |          |          |          |          |
| **FGSMAdv**   |     71.4 | **93.5** |      0.3 |       21 |
| **Fast**      | **84.7** |     55.0 | **45.3** |       21 |
| **Free**      |     68.1 |     42.7 |     39.6 |   **14** |
| **GradAlign** |     84.5 |     53.4 |     44.0 |       95 |
| _Mutli-step_  |          |          |          |          |
| **PGDAdv**    | **81.0** | **56.2** |     50.7 |   **87** |
| **TRADES**    |     79.5 |     54.5 |     50.7 |      101 |
| **MART**      |     75.8 |     55.7 | **52.5** |       95 |



#### Stepwise

| PRN18         | Standard | FGSM |  PGD | Time (m) |
| ------------- | -------: | ---: | ---: | -------: |
| **Base**      |     93.7 | 12.9 |  0.0 |       46 |
|               |          |      |      |          |
| **FGSMAdv**   |     72.1 | 45.2 | 38.9 |          |
| **Fast**      |     77.2 | 45.6 | 37.9 |          |
| **Free**      |     73.9 | 44.4 | 39.0 |          |
| **GradAlign** |     78.2 | 44.3 | 35.5 |          |
|               |          |      |      |          |
| **PGDAdv**    |     71.9 | 45.4 | 40.9 |          |
| **TRADES**    |     69.1 | 45.5 | 42.5 |          |
| **MART**      |     75.5 | 52.5 | 46.8 |          |

