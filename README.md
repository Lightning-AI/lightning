<div align="center">

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ptl_banner.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

**The deep learning framework to pretrain and finetune AI models.**

**Serving models?** Use [LitServe](https://github.com/Lightning-AI/litserve?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) to build custom inference servers in pure Python.

______________________________________________________________________

<p align="center">
    <a href="#quick-start" style="margin: 0 10px;">Quick start</a> •
  <a href="#examples">Examples</a> •
  <a href="#why-pytorch-lightning">PyTorch Lightning</a> •
  <a href="#lightning-fabric-expert-control">Fabric</a> •
  <a href="https://lightning.ai/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme">Lightning Cloud</a> •   
  <a href="#community">Community</a> •
  <a href="https://pytorch-lightning.readthedocs.io/en/stable/">Docs</a>
</p>

<!-- DO NOT ADD CONDA DOWNLOADS... README CHANGES MUST BE APPROVED BY EDEN OR WILL -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![PyPI Status](https://badge.fury.io/py/pytorch-lightning.svg)](https://badge.fury.io/py/pytorch-lightning)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-lightning)](https://pepy.tech/project/pytorch-lightning)
[![Conda](https://img.shields.io/conda/v/conda-forge/lightning?label=conda&color=success)](https://anaconda.org/conda-forge/lightning)
[![codecov](https://codecov.io/gh/Lightning-AI/pytorch-lightning/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/pytorch-lightning)

[![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/lightning-ai/lightning)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/pytorch-lightning/blob/master/LICENSE)

<!--
[![CodeFactor](https://www.codefactor.io/repository/github/Lightning-AI/lightning/badge)](https://www.codefactor.io/repository/github/Lightning-AI/lightning)
-->

</div>

<div align="center">
  
<p align="center">

&nbsp;

<a target="_blank" href="https://lightning.ai/docs/pytorch/latest/starter/introduction.html?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme#define-a-lightningmodule">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</p>

</div>

&nbsp;

<a id="why-pytorch-lightning"></a>
# Why PyTorch Lightning?   
Training models in plain PyTorch requires writing and maintaining a lot of repetitive engineering code. Handling backpropagation, mixed precision, multi-GPU, and distributed training is error-prone and often reimplemented for every project. PyTorch Lightning organizes PyTorch code to automate this infrastructure while keeping full control over your model logic. You write the science. Lightning handles the engineering, and scales from CPU to multi-node GPUs without changing your core code. PyTorch experts can still opt into [expert-level control](#lightning-fabric-expert-control).   

Fun analogy: If PyTorch is Javascript, PyTorch Lightning is ReactJS or NextJS.

# Looking for GPUs?
[Lightning Cloud](https://lightning.ai/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) is the easiest way to run PyTorch Lightning without managing infrastructure. Start training with one command and get GPUs, autoscaling, monitoring, and a free tier. No cloud setup required.

You can also run PyTorch Lightning on your own hardware or cloud.

# Lightning has 2 core packages

[PyTorch Lightning: Train and deploy PyTorch at scale](#why-pytorch-lightning).
<br/>
[Lightning Fabric: Expert control](#lightning-fabric-expert-control).

Lightning gives you granular control over how much abstraction you want to add over PyTorch.

<div align="center">
    <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/continuum.png" width="80%">
</div>

&nbsp;

# Quick start
Install Lightning:

```bash
pip install lightning
```

<!-- following section will be skipped from PyPI description -->

<details>
  <summary>Advanced install options</summary>
    <!-- following section will be skipped from PyPI description -->

#### Install with optional dependencies

```bash
pip install lightning['extra']
```

#### Conda

```bash
conda install lightning -c conda-forge
```

#### Install stable version

Install future release from the source

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
```

#### Install bleeding-edge

Install nightly from the source (no guarantees)

```bash
pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U
```

or from testing PyPI

```bash
pip install -iU https://test.pypi.org/simple/ pytorch-lightning
```

</details>
<!-- end skipping PyPI description -->

### PyTorch Lightning example
Define the training workflow. Here's a toy example ([explore real examples](https://lightning.ai/environments?view=public&section=featured&query=pytorch+lightning&utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme)):

```python
# main.py
# ! pip install torchvision
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# -------------------
# Step 2: Define data
# -------------------
dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [55000, 5000])

# -------------------
# Step 3: Train
# -------------------
autoencoder = LitAutoEncoder()
trainer = L.Trainer()
trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))
```

Run the model on your terminal

```bash
pip install torchvision
python main.py
```

&nbsp;


# Convert from PyTorch to PyTorch Lightning

PyTorch Lightning is just organized PyTorch - Lightning disentangles PyTorch code to decouple the science from the engineering.

![PT to PL](docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif)

&nbsp;

----

### Examples
Explore various types of training possible with PyTorch Lightning. Pretrain and finetune ANY kind of model to perform ANY task like classification, segmentation, summarization and more:    

| Task | Description | Run |
|------|--------------|-----|
| [Hello world](https://lightning.ai/lightning-ai/studios/pytorch-lightning-hello-world?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Pretrain - Hello world example | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/pytorch-lightning-hello-world?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Image classification](https://lightning.ai/lightning-ai/studios/image-classification-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - ResNet-34 model to classify images of cars | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/image-classification-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Image segmentation](https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - ResNet-50 model to segment images | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Object detection](https://lightning.ai/lightning-ai/studios/object-detection-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - Faster R-CNN model to detect objects | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/object-detection-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Text classification](https://lightning.ai/lightning-ai/studios/text-classification-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - text classifier (BERT model) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/text-classification-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Text summarization](https://lightning.ai/lightning-ai/studios/text-summarization-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - text summarization (Hugging Face transformer model) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/text-summarization-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Audio generation](https://lightning.ai/lightning-ai/studios/finetune-a-personal-ai-music-generator?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - audio generator (transformer model) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/finetune-a-personal-ai-music-generator?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [LLM finetuning](https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Finetune - LLM (Meta Llama 3.1 8B) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/finetune-an-llm-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Image generation](https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Pretrain - Image generator (diffusion model) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Recommendation system](https://lightning.ai/lightning-ai/studios/recommendation-system-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Train - recommendation system (factorization and embedding) | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/recommendation-system-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |
| [Time-series forecasting](https://lightning.ai/lightning-ai/studios/time-series-forecasting-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) | Train - Time-series forecasting with LSTM | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/time-series-forecasting-with-pytorch-lightning?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a> |


______________________________________________________________________

## Advanced features

Lightning has over [40+ advanced features](https://lightning.ai/docs/pytorch/stable/common/trainer.html?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme#trainer-flags)
designed for professional AI research at scale.

Here are some examples:

<div align="center">
    <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/features_2.jpg" max-height="600px">
  </div>

<details>
  <summary>Train on 1000s of GPUs without code changes</summary>

```python
# 8 GPUs
# no code changes needed
trainer = Trainer(accelerator="gpu", devices=8)

# 256 GPUs
trainer = Trainer(accelerator="gpu", devices=8, num_nodes=32)
```

</details>

<details>
  <summary>Train on other accelerators like TPUs without code changes</summary>

```python
# no code changes needed
trainer = Trainer(accelerator="tpu", devices=8)
```

</details>

<details>
  <summary>16-bit precision</summary>

```python
# no code changes needed
trainer = Trainer(precision=16)
```

</details>

<details>
  <summary>Experiment managers</summary>

```python
from lightning import loggers

# litlogger
trainer = Trainer(logger=LitLogger())

# tensorboard
trainer = Trainer(logger=TensorBoardLogger("logs/"))

# weights and biases
trainer = Trainer(logger=loggers.WandbLogger())

# comet
trainer = Trainer(logger=loggers.CometLogger())

# mlflow
trainer = Trainer(logger=loggers.MLFlowLogger())

# ... and dozens more
```

</details>

<details>

<summary>Early Stopping</summary>

```python
es = EarlyStopping(monitor="val_loss")
trainer = Trainer(callbacks=[es])
```

</details>

<details>
  <summary>Checkpointing</summary>

```python
checkpointing = ModelCheckpoint(monitor="val_loss")
trainer = Trainer(callbacks=[checkpointing])
```

</details>

<details>
  <summary>Export to torchscript (JIT) (production use)</summary>

```python
# torchscript
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
```

</details>

<details>
  <summary>Export to ONNX (production use)</summary>

```python
# onnx
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
    autoencoder = LitAutoEncoder()
    input_sample = torch.randn((1, 64))
    autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    os.path.isfile(tmpfile.name)
```

</details>

______________________________________________________________________

## Advantages over unstructured PyTorch

- Models become hardware agnostic
- Code is clear to read because engineering code is abstracted away
- Easier to reproduce
- Make fewer mistakes because lightning handles the tricky engineering
- Keeps all the flexibility (LightningModules are still PyTorch modules), but removes a ton of boilerplate
- Lightning has dozens of integrations with popular machine learning tools.
- [Tested rigorously with every new PR](https://github.com/Lightning-AI/lightning/tree/master/tests). We test every combination of PyTorch and Python supported versions, every OS, multi GPUs and even TPUs.
- Minimal running speed overhead (about 300 ms per epoch compared with pure PyTorch).

______________________________________________________________________

<div align="center">
    <a href="https://lightning.ai/docs/pytorch/stable/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme">Read the PyTorch Lightning docs</a>
</div>

______________________________________________________________________

&nbsp;
&nbsp;

# Lightning Fabric: Expert control

Run on any device at any scale with expert-level control over PyTorch training loop and scaling strategy. You can even write your own Trainer.

Fabric is designed for the most complex models like foundation model scaling, LLMs, diffusion, transformers, reinforcement learning, active learning. Of any size.

<table>
<tr>
<th>What to change</th>
<th>Resulting Fabric Code (copy me!)</th>
</tr>
<tr>
<td>
<sub>

```diff
+ import lightning as L
  import torch; import torchvision as tv

 dataset = tv.datasets.CIFAR10("data", download=True,
                               train=True,
                               transform=tv.transforms.ToTensor())

+ fabric = L.Fabric()
+ fabric.launch()

  model = tv.models.resnet18()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
- device = "cuda" if torch.cuda.is_available() else "cpu"
- model.to(device)
+ model, optimizer = fabric.setup(model, optimizer)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
+ dataloader = fabric.setup_dataloaders(dataloader)

  model.train()
  num_epochs = 10
  for epoch in range(num_epochs):
      for batch in dataloader:
          inputs, labels = batch
-         inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = torch.nn.functional.cross_entropy(outputs, labels)
-         loss.backward()
+         fabric.backward(loss)
          optimizer.step()
          print(loss.data)
```

</sub>
<td>
<sub>

```Python
import lightning as L
import torch; import torchvision as tv

dataset = tv.datasets.CIFAR10("data", download=True,
                              train=True,
                              transform=tv.transforms.ToTensor())

fabric = L.Fabric()
fabric.launch()

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model, optimizer = fabric.setup(model, optimizer)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        print(loss.data)
```

</sub>
</td>
</tr>
</table>

## Key features

<details>
  <summary>Easily switch from running on CPU to GPU (Apple Silicon, CUDA, …), TPU, multi-GPU or even multi-node training</summary>

```python
# Use your available hardware
# no code changes needed
fabric = Fabric()

# Run on GPUs (CUDA or MPS)
fabric = Fabric(accelerator="gpu")

# 8 GPUs
fabric = Fabric(accelerator="gpu", devices=8)

# 256 GPUs, multi-node
fabric = Fabric(accelerator="gpu", devices=8, num_nodes=32)

# Run on TPUs
fabric = Fabric(accelerator="tpu")
```

</details>

<details>
  <summary>Use state-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed) and mixed precision out of the box</summary>

```python
# Use state-of-the-art distributed training techniques
fabric = Fabric(strategy="ddp")
fabric = Fabric(strategy="deepspeed")
fabric = Fabric(strategy="fsdp")

# Switch the precision
fabric = Fabric(precision="16-mixed")
fabric = Fabric(precision="64")
```

</details>

<details>
  <summary>All the device logic boilerplate is handled for you</summary>

```diff
  # no more of this!
- model.to(device)
- batch.to(device)
```

</details>

<details>
  <summary>Build your own custom Trainer using Fabric primitives for training checkpointing, logging, and more</summary>

```python
import lightning as L


class MyCustomTrainer:
    def __init__(self, accelerator="auto", strategy="auto", devices="auto", precision="32-true"):
        self.fabric = L.Fabric(accelerator=accelerator, strategy=strategy, devices=devices, precision=precision)

    def fit(self, model, optimizer, dataloader, max_epochs):
        self.fabric.launch()

        model, optimizer = self.fabric.setup(model, optimizer)
        dataloader = self.fabric.setup_dataloaders(dataloader)
        model.train()

        for epoch in range(max_epochs):
            for batch in dataloader:
                input, target = batch
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                self.fabric.backward(loss)
                optimizer.step()
```

You can find a more extensive example in our [examples](examples/fabric/build_your_own_trainer)

</details>

______________________________________________________________________

<div align="center">
    <a href="https://lightning.ai/docs/fabric/stable/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme">Read the Lightning Fabric docs</a>
</div>

______________________________________________________________________

&nbsp;
&nbsp;

## Examples

###### Self-supervised Learning

- [CPC transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#cpc-transforms)
- [Moco v2 transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#moco-v2-transforms)
- [SimCLR transforms](https://lightning-bolts.readthedocs.io/en/stable/transforms/self_supervised.html#simclr-transforms)

###### Convolutional Architectures

- [GPT-2](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#gpt-2)
- [UNet](https://lightning-bolts.readthedocs.io/en/stable/models/convolutional.html#unet)

###### Reinforcement Learning

- [DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#dqn-loss)
- [Double DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#double-dqn-loss)
- [Per DQN Loss](https://lightning-bolts.readthedocs.io/en/stable/losses.html#per-dqn-loss)

###### GANs

- [Basic GAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#basic-gan)
- [DCGAN](https://lightning-bolts.readthedocs.io/en/stable/models/gans.html#dcgan)

###### Classic ML

- [Logistic Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#logistic-regression)
- [Linear Regression](https://lightning-bolts.readthedocs.io/en/stable/models/classic_ml.html#linear-regression)

&nbsp;
&nbsp;

## Continuous Integration

Lightning is rigorously tested across multiple CPUs, GPUs and TPUs and against major Python and PyTorch versions.

###### \*Codecov is > 90%+ but build delays may show less

<details>
  <summary>Current build statuses</summary>

<center>

|       System / PyTorch ver.        | 1.13                                                                                                                                                                                                                            | 2.0                                                                                                                                                                                                                             |                                                                                                               2.1                                                                                                               |
| :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        Linux py3.9 \[GPUs\]        |  |  | [![Build Status](https://dev.azure.com/Lightning-AI/lightning/_apis/build/status%2Fpytorch-lightning%20%28GPUs%29?branchName=master)](https://dev.azure.com/Lightning-AI/lightning/_build/latest?definitionId=24&branchName=master) |
|  Linux (multiple Python versions)  | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 |
|   OSX (multiple Python versions)   | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 |
| Windows (multiple Python versions) | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 | [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                                 |                 [![Test PyTorch](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml/badge.svg)](https://github.com/Lightning-AI/lightning/actions/workflows/ci-tests-pytorch.yml)                 |

</center>
</details>

&nbsp;
&nbsp;

## Community

The lightning community is maintained by

- [10+ core contributors](https://lightning.ai/docs/pytorch/latest/community/governance.html) who are all a mix of professional engineers, Research Scientists, and Ph.D. students from top AI labs.
- 800+ community contributors.

Want to help us build Lightning and reduce boilerplate for thousands of researchers? [Learn how to make your first contribution here](https://lightning.ai/docs/pytorch/stable/generated/CONTRIBUTING.html?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme)

Lightning is also part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/) which requires projects to have solid testing, documentation and support.

### Asking for help

If you have any questions please:

1. [Read the docs](https://lightning.ai/docs?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme).
1. [Search through existing Discussions](https://github.com/Lightning-AI/lightning/discussions), or [add a new question](https://github.com/Lightning-AI/lightning/discussions/new)
1. [Join our discord](https://discord.com/invite/tfXFetEZxv).
