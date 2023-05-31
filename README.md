This is the code-implementation for the paper [Collaborative learning via Prediction Consensus](https://arxiv.org/abs/2305.18497)

## Environment
```bash
conda env create -f env.yml
conda activate collab-consensus
```
## Data generation

Supported datasets:
- `Cifar10/100`
- [`FedISIC2019`](https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/README.md)

Go to `datasets` folder. For `cifar10/100`, dirichlet ($\alpha$ = a) distribution is used to create statistical heterogeniety among agents. Bigger $\alpha$ means less non-iidness. For `FedISIC2019` dataset, each split corresponds to data collected by different hospitals. 

```python
python3 generate_cifar100.py --niid --partition dir --alpha [a] --n_clients [n] --refgen 
```

## Model Training
By default, models are trained with 50 gloabl rounds and 5 local rounds each. 
Here we offer different combinations for training:
- Setting: 
    - Normal 
    - Clients are assigned with two different model architectures
    - Clients with low quality data involved
- Trust weight update:
    - dynamic
    - static (fixed as first trust calculation throughout training)
    - naive (fixed as 1/N each entry throughout training)

```python
python3 main.py -ds [Cifar10/Cifar100/] -ncl [10] -lam [0.5/0.0] -setting [normal/2sets/evil] -trust [dynamic/static/naive]
```
