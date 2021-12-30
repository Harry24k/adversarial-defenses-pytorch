# Adversarial-Defenses-PyTorch (*Under Reconstruction*)

<p>
  <a href="https://github.com/Harry24k/adversarial-defenses-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/adversarial-defenses-pytorch?&color=brightgreen" /></a>
  <a href="https://pypi.org/project/torchdefenses/"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchdefenses.svg?&color=orange" /></a>
  <a href="https://github.com/Harry24k/adversarial-torchdefenses-pytorch/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/Harry24k/adversarial-torchdefenses-pytorch.svg?&color=blue" /></a>

[Torchdefenses] is a PyTorch library that provides *adversarial defenses* to obtain robust model against adversarial attacks. It contains *PyTorch Lightning-like* interface and functions that make it easier for PyTorch users to implement adversarial defenses.

## How to use?

```python
import torchdefenses as td
rmodel = td.RobModel(model, n_classes=10, 
                     normalize={'mean':[0.4914, 0.4822, 0.4465], 'std':[0.2023, 0.1994, 0.2010]})
```

<details><summary>Easy training</summary><p>

```python
import torchdefenses.trainer as tr
trainer = tr.Standard(rmodel)
trainer.record_rob(train_loader, val_loader, eps=0.3, alpha=0.1, steps=5, std=0.1)
trainer.fit(train_loader=train_loader, max_epoch=10, optimizer="SGD(lr=0.01)",
            scheduler="Step([100, 105], 0.1)", scheduler_type="Epoch",
            record_type="Epoch", save_type="Epoch",
            save_path="./_temp/"+"sample", save_overwrite=True)
```
</p></details>

<details><summary>Supporting Multi-GPU</summary><p>


```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # Possible GPUS

model = td.utils.load_model(model_name="ResNet18", n_classes=10).cuda() # Load model
model = torch.nn.DataParallel(model) # Parallelize

rmodel = td.RobModel(model, n_classes=10, 
                  normalize={'mean':[0.4914, 0.4822, 0.4465], 'std':[0.2023, 0.1994, 0.2010]}) # Wrap it with RobModel
trainer = ... # Define trainer
trainer.fit(..) # Training start
```

</p></details>

<details><summary>Recording, Saving and Visualizing</summary><p>

```python
trainer.save_all("./_temp/"+"sample", overwrite=True)
trainer.rm.plot(title="A", xlabel="Epoch", ylabel="Accuracy",
                figsize=(6, 4),
                x_key='Epoch',
                y_keys=['Clean(Tr)', 'FGSM(Tr)', 'PGD(Tr)', 'GN(Tr)',
                        'Clean(Val)', 'FGSM(Val)', 'PGD(Val)', 'GN(Val)'],
                ylim=(-10, 110),
                colors=['k', '#D81B60', '#1E88E5', '#004D40']*2,
                labels=['Clean', 'FGSM', 'PGD', 'GN', '', '', '', ''],
                linestyles=['-', '-', '-', '-', '--', '--', '--', '--'],
               )
```
</p></details>

<details><summary>Easy evaluation</summary><p>

```python
rmodel.eval_accuracy(test_loader)
rmodel.eval_rob_accuracy_pgd(test_loader, eps=1, alpha=0.1,
                             steps=10, random_start=True, restart_num=1)
```
</p></details>

<details><summary>Useful functions</summary><p>

```python
from torchdefenses.utils import fix_randomness, fix_gpu
fix_randomness(0)
fix_gpu(0)
```
</p></details>

## How to customize?

