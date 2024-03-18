#!/bin/bash

#SBATCH -o run_logs/R-outFile_%j.txt
#SBATCH -p publicgpu          # Partition public avec des GPU
#SBATCH -N 1                  # 1 nœud
# --exclusive           # Le nœud sera entièrement dédié à notre job, pas de partage de ressources
#SBATCH -t 08:00:00           # Le job sera tué au bout de 4h
#SBATCH --gres=gpu:1          # 1 GPU par nœud
#SBATCH --constraint=gputc    # Nœuds GPU tensor cores
# --mail-type=END       # Réception d'un mail à la fin du job
# --mail-user=q.christoffel@unistra.fr

export TF_CPP_MIN_LOG_LEVEL=3

if [ $1 == "slurm" ];
then
    echo "Running on slurm"
    echo ""
    echo "Command executed:"
    echo "sbatch bash_scripts/launch.sh $@"
    echo ""
    echo "Working on node : $SLURM_JOB_NODELIST"
    echo "Number of CPU on this node: $SLURM_CPUS_ON_NODE"

    module load python/Anaconda3-2019
    source activate tf

    # export CUDA_VISIBLE_DEVICES=0

    hostname
    echo ""
elif [ $1 == "local" ];
then 
    echo "Running locally"
    echo ""
    echo "Command executed:"
    echo "bash bash_scripts/launch.sh $@"
else
    echo "Error: first argument must be either 'slurm' or 'local'"
    exit 1
fi

shift 1

python3 train_model.py $@ 

# python3 train_model.py     --save_path results/tests_tinyIm     --dataset tiny_imagenet --config 4     --epochs 200     --lr 0.01     --model standard_vgg32     --loss crossentropy     --randaug_n 1 --randaug_m 9     --image_size 64     --nb_features 1     --anchor_multiplier 10     --max_dist 12.649     --verbose 2     --osr_score max     --batch_size 128     --fc_end   --scheduler cosine  --summary $@

# python3 train_model_softmax.py \
    # --dataset cifar10 \
    # --epochs 600 \
    # --save_path "networks/cifar10/softmax/" \
    # --prefix new_600ep_coslr_randAug_difSplit_ \
    # --batch_size 128 \
    # --verbose 1 \
    # --nosplit_train_val \
## pass all the arguments to the script, override default values above

exit $?
