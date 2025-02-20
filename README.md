


# Fundust image classification

[![GitHub license](https://img.shields.io/github/license/用户名/仓库名)](https://github.com/用户名/仓库名/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 🛠️ Install dependencies
```python
conda create -n PDM python=3.11.7
conda activate PDM
pip install -r requirements.txt
```

## 📁 Data preparation
download GastroVision or Ulcerative or OCT2017 or chest_xray or medMNIST or ISIC datasets



## 🚀 Train

```python
python Source/Train.py  --dataset BreastMNIST  --model resnet50 --CCE_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
```


