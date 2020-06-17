# DAML

Source code for the ACL 2019 paper entitled "Domain Adaptive Dialog Generation via Meta Learning" by Kun Qian and Zhou Yu https://arxiv.org/abs/1906.03520


```
@article{qian2019domain,
  title={Domain Adaptive Dialog Generation via Meta Learning},
  author={Qian, Kun and Yu, Zhou},
  journal={arXiv preprint arXiv:1906.03520},
  year={2019}
}
```
## Simulated Data Generation
Please download the code here: https://github.com/qbetterk/SimDial
```
git clone https://github.com/qbetterk/SimDial.git
cd SimDial
python multiple_domains.py
```

## Training with default parameters

```
python model.py
```

(optional: configuring hyperparameters with cmdline)

```
python model.py -mode train_maml -model tsdf-camrest -cfg lr=0.003 batch_size=32
```

## Testing

```
python model.py -mode test_maml -model tsdf-camrest
```

or test on new domains (e.g. movie domain):
```
bash run_movie.sh
```


## Before running
1. Install required python packages. We used pytorch 0.3.0 and python 3.6 under Linux operating system. 
```
pip install -r requirements.txt
```
2. Make directories under PROJECT_ROOT.
```
mkdir vocab
mkdir log
mkdir results
mkdir models
mkdir sheets
```

3. Download pretrained Glove word vectors and place them in PROJECT_ROOT/data/glove.
