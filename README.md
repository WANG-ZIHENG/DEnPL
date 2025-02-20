


# PDM

[![GitHub license](https://img.shields.io/github/license/用户名/仓库名)](https://github.com/用户名/仓库名/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 🛠️ Install dependencies
```python
conda create -n PDM python=3.8.10
conda activate PDM
pip install -r requirements.txt
```

## 📁 Data preparation

## 

| Datasets       | Modalities         | Classes | Download                                                |
| -------------- | ------------------ | ------- | ------------------------------------------------------- |
| GastroVision   | Endoscopy          | 27      | [link](https://osf.io/84e7f/)                           |
| LIMUC          | Endoscopy          | 4       | [link](https://zenodo.org/records/5827695#.Yi8GJ3pByUk) |
| ISIC 2019      | Dermatoscopy       | 8       | [link](https://challenge.isic-archive.com/data/#2019)   |
| BloodMNIST     | Digital Microscopy | 8       | [link](https://zenodo.org/records/10519652)             |
| OrganCMNIST    | CT                 | 11      | [link](https://zenodo.org/records/10519652)             |
| BreastMNIST    | Ultrasound         | 2       | [link](https://zenodo.org/records/10519652)             |
| DermaMNIST     | Dermatoscopy       | 7       | [link](https://zenodo.org/records/10519652)             |
| PneumoniaMNIST | X-ray              | 2       | [link](https://zenodo.org/records/10519652)             |
| BUS-BRA        | Ultrasound         | 2       | [link](https://zenodo.org/records/8231412)              |





## 🚀 Train

```python
python Source/Train.py  --dataset BreastMNIST  --model resnet50 --CCE_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
```
The complete code will be made publicly available after the paper is accepted. **Coming Soon.**

