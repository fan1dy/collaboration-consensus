source ~/.bashrc
conda activate py38
cd /mlodata1/dongyang/codes/edic-dongyang/

# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 1 -ncl 100 -trust dynamic

# python3 main.py -sim true_label -setting normal -ds Cifar100 -metric acc -seed 1 -ncl 100 -trust static

# #normal - no collab
# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 1 -ncl 100 -lam 0

#normal - naive trust
# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 1 -ncl 100 -trust naive

# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 2 -expno 1 -ncl 100 

# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 2 -expno 1 -ncl 100 -trust static

# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 2 -expno 1 -ncl 100 -lam 0

# python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 2 -expno 1 -ncl 100 -trust naive


python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 3 -expno 2 -ncl 100 -respath /mlodata1/dongyang/results/res_april/
 
python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 3 -expno 2 -ncl 100 -trust static -respath /mlodata1/dongyang/results/res_april/

python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 3 -expno 2 -ncl 100 -lam 0 -respath /mlodata1/dongyang/results/res_april/

python3 main.py -sim cosine -setting normal -ds Cifar100 -metric acc -seed 3 -expno 2 -ncl 100 -trust naive -respath /mlodata1/dongyang/results/res_april/