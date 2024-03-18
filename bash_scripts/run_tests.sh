#!/bin/bash

################################################################################
# This script is used to test the different components of the code.
# But it only tests if the code is running, not if the results are correct.
################################################################################

export CUDA_VISIBLE_DEVICES=0

# python3 train_model.py --dataset mnist --verbose 2 --epochs 50 --save_path networks/mnist --prefix test_mnist_  > run_mnist.log 2>&1
# python3 train_model.py --dataset cifar10 --verbose 2 --epochs 150 --save_path networks/cifar10 --prefix test_cifar10_ > run_cifar10.log 2>&1
# python3 train_model.py --dataset svhn_cropped --verbose 2 --epochs 100 --save_path networks/svhn_cropped --prefix test_svhn_cropped_ > run_svhn_cropped.log 2>&1

start_time=$(date +%s)

# echo "Testing schedulers on fake data"
# error_count=0
# for scheduler in constant cac cosine cosine_restart_warmup; do
#     elapsed_time=$(($(date +%s) - $start_time))
#     echo "-- $elapsed_time: Running scheduler $scheduler"
#     mkdir -p /tmp/T-schedulers/

#     # Testing on a consequent number of epochs to see if the scheduler is "working" (not crashing)
#     bash bash_scripts/launch.sh local               \
#         --flagfile default_flags/cifar10_flags.txt  \
#         --dataset fake                              \
#         --scheduler $scheduler                      \
#         --save_path /tmp/T-schedulers/              \
#         --prefix ${scheduler}_                       \
#         --epochs 100 > /tmp/T-schedulers/${scheduler}_fake.out 2>&1

#     if [ $? -ne 0 ]; then
#         error_count=$((error_count+1))
#         echo -e "\t\e[31m Error: scheduler $scheduler failed\e[0m"
#         echo -e "\t\e[31m Re-run using : bash bash_scripts/launch.sh local --flagfile /tmp/T-schedulers/${scheduler}_flags.txt \e[0m"
#         tail -n 10 /tmp/T-schedulers/${scheduler}_fake.out
#     else
#         echo -e "\t\e[32m Success\e[0m"
#         rm /tmp/T-schedulers/${scheduler}_fake.out
#     fi

#     if [ $error_count -eq 0 ]; then
#         rm -r /tmp/T-schedulers/
#     fi
# done

echo ""
echo "Testing losses on CIFAR10"
error_count=0
# for loss in compact_hypersphere new; do 
for loss in crossentropy cac difair compact_hypersphere new_v0 new_v1; do
    task_time=$(date +%s)
    elapsed_time=$(($(date +%s) - $start_time))

    prefix=${loss}-${task_time}_

    echo "-- $elapsed_time: Running loss $loss"
    mkdir -p /tmp/T-losses/

    bash bash_scripts/launch.sh local               \
        --flagfile default_flags/cifar10_flags.txt  \
        --loss $loss                                \
        --save_path /tmp/T-losses/                  \
        --prefix ${prefix}                          \
        --epochs 3 > /tmp/T-losses/${prefix}_cifar10.out 2>&1 

    if [ $? -ne 0 ]; then
        error_count=$((error_count+1))
        echo -e "\t\e[31m Error: loss $loss failed\e[0m"
        echo -e "\t\e[31m Re-run using : bash bash_scripts/launch.sh local --flagfile /tmp/T-losses/${prefix}_flags.txt \e[0m"
        tail -n 10 /tmp/T-losses/${prefix}_cifar10.out
    else
        echo -e "\t\e[32m Success\e[0m"
        rm /tmp/T-losses/${prefix}_cifar10.out
    fi

    if [ ${error_count} -eq 0 ]; then
        rm -r /tmp/T-losses/
    fi
done

echo ""
echo "Testing different datasets with validation split"
error_count=0
for dataset in "svhn" "cifar10" "cifar+10" "cifar+50"; do
    task_time=$(date +%s)
    elapsed_time=$(($(date +%s) - $start_time))

    prefix=${dataset}-${task_time}_

    echo "-- $elapsed_time: Running dataset $dataset"
    mkdir -p /tmp/T-datasets/

    bash bash_scripts/launch.sh local                   \
        --flagfile default_flags/${dataset}_flags.txt   \
        --save_path /tmp/T-datasets/                    \
        --prefix ${prefix}                              \
        --split_train_val                               \
        --epochs 3 > /tmp/T-datasets/${prefix}.out 2>&1

    if [ $? -ne 0 ]; then
        error_count=$((error_count+1))
        echo -e "\t\e[31m Error: dataset $dataset failed\e[0m"
        echo -e "\t\e[31m Re-run using : bash bash_scripts/launch.sh local --flagfile /tmp/T-datasets/${prefix}_flags.txt \e[0m"
        tail -n 10 /tmp/T-datasets/${prefix}.out
    else
        echo -e "\t\e[32m Success\e[0m"
        rm /tmp/T-datasets/${prefix}.out
    fi

    if [ ${error_count} -eq 0 ]; then
        rm -r /tmp/T-datasets/
    fi
done