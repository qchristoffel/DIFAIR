#!/bin/bash

#SBATCH -o run_logs/R-outFile_%j.txt
#SBATCH -p publicgpu          # Partition public avec des GPU
#SBATCH -N 1                  # 1 nœud
#SBATCH --exclusive           # Le nœud sera entièrement dédié à notre job, pas de partage de ressources
#SBATCH -t 08:00:00           # Le job sera tué au bout de 8h
#SBATCH --gres=gpu:2          # 2 GPU par nœud
#SBATCH --constraint=gputc    # Nœuds GPU tensor cores
#SBATCH --mail-type=END       # Réception d'un mail à la fin du job
#SBATCH --mail-user=q.christoffel@unistra.fr

export TF_CPP_MIN_LOG_LEVEL=2

if [ $1 == "slurm" ];
then
    echo sbatch bash_scripts/execute_bash_file.sh $@
    echo ""
    echo "Number of CPU on this node: $SLURM_CPUS_ON_NODE"
    module load python/Anaconda3-2019
    source activate tf

    hostname
    echo ""
elif [ $1 == "local" ];
then 
    echo "Running locally"
    echo ""
else
    echo "Error: first argument must be either 'slurm' or 'local'"
    exit 1
fi

shift 1

bash $@