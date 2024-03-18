import math

import tensorflow as tf
from tensorflow import keras


def get_scheduler(args, nb_batches):
    if args.scheduler == "constant":
        return args.lr

    elif args.scheduler == "cac":
        tf.print("Using CAC scheduler")
        return LRScheduleCAC(
            initial_learning_rate=args.lr,
            nb_batches=nb_batches,
            epoch_swap=args.epoch_swap,
        )

    elif args.scheduler == "cosine":
        tf.print("Using cosine scheduler")

        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.epochs * nb_batches,
            alpha=args.lr * 1e-3,
        )

    elif args.scheduler == "cosine_restart_warmup":
        tf.print("Using cosine_restart_warmup scheduler")

        try:
            num_restarts = args.num_restarts
        except:
            print("Warning: Num restarts not specified...using 2")
            num_restarts = 2

        return CosineDecayRestartWithWarmup(
            nb_batches=nb_batches,
            initial_learning_rate=args.lr,
            first_decay_steps=int(args.epochs / (num_restarts + 1)),
            t_mul=1.0,
            m_mul=1.0,
            alpha=args.lr * 1e-3,
            warmup_steps=10,
        )

    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented")


class LRScheduleCAC(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, nb_batches, epoch_swap):
        super(LRScheduleCAC, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.nb_batches = nb_batches
        self.epoch_swap = epoch_swap
        self.lr_schedule = self.lr_schedule_cac(nb_batches, epoch_swap)

    def lr_schedule_cac(self, nb_batches, epoch_swap):
        """Used to replicate the learning rate schedule from the CAC paper"""

        def _lr_schedule_step(step, lr):
            epoch = step // nb_batches

            def reduce_lr():
                return 0.001

            lr = tf.cond(epoch > epoch_swap, reduce_lr, lambda: lr)
            return lr

        return _lr_schedule_step

    def __call__(self, step):
        return self.lr_schedule(step, self.initial_learning_rate)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "nb_batches": self.nb_batches,
            "epoch_swap": self.epoch_swap,
        }

    def set_config(self, config):
        self.initial_learning_rate = config["initial_learning_rate"]
        self.nb_batches = config["nb_batches"]
        self.epoch_swap = config["epoch_swap"]


# linearly augment learning rate from 0 to initial_learning_rate over warmup_steps
# then return result from CosineDecayRestarts
class CosineDecayRestartWithWarmup(tf.keras.optimizers.schedules.CosineDecayRestarts):
    def __init__(
        self,
        nb_batches,
        initial_learning_rate,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        warmup_steps=0,
        name=None,
    ):
        super(CosineDecayRestartWithWarmup, self).__init__(
            initial_learning_rate, first_decay_steps, t_mul, m_mul, alpha, name
        )

        self.nb_batches = nb_batches
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps

        # follow implementation of https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/utils/schedulers.py
        target_lr = (
            self.alpha
            + (self.initial_learning_rate - self.alpha)
            * (1 + math.cos(math.pi * self.warmup_steps / self.first_decay_steps))
            / 2
        )

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.alpha) / self.warmup_steps
        print("Linear step:", linear_step)
        self.warmup_lrs = [
            self.alpha + linear_step * (n + 1) for n in range(self.warmup_steps)
        ]
        self.warmup_lrs.insert(
            0, self.alpha
        )  # to be consistent with the pytorch implementation
        self.warmup_lrs = tf.constant(self.warmup_lrs, dtype=tf.float32)
        print("Warmup lrs:", self.warmup_lrs)

    def __call__(self, step):
        epoch = step // self.nb_batches

        def warmup_lr():
            return self.warmup_lrs[epoch]

        def cosine_lr():
            return super(CosineDecayRestartWithWarmup, self).__call__(epoch - 1)

        tf.summary.scalar(
            "learning_rate", tf.cond(epoch <= self.warmup_steps, warmup_lr, cosine_lr)
        )
        return tf.cond(epoch <= self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "nb_batches": self.nb_batches,
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }

    def set_config(self, config):
        self.nb_batches = config["nb_batches"]
        self.initial_learning_rate = config["initial_learning_rate"]
        self.first_decay_steps = config["first_decay_steps"]
        self.t_mul = config["t_mul"]
        self.m_mul = config["m_mul"]
        self.alpha = config["alpha"]
        self.warmup_steps = config["warmup_steps"]
        self.name = config["name"]
