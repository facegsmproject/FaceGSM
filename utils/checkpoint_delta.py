import os
import tensorflow as tf
from utils.error_handling import show_info


def save_checkpoint(deltaUpdated):
    show_info("Saving Checkpoint...")
    checkpoint = tf.train.Checkpoint(delta=deltaUpdated)
    manager = tf.train.CheckpointManager(checkpoint, "./checkpoints", max_to_keep=3)
    manager.save()
    return


def load_checkpoint(deltaUpdated):
    checkpoint = tf.train.Checkpoint(delta=deltaUpdated)
    manager = tf.train.CheckpointManager(checkpoint, "./checkpoints", max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print("Restored from {}".format(manager.latest_checkpoint))
        return deltaUpdated
    else:
        print("None.")


def signal_handler(sig, frame, delta):
    print(
        "\nDo you want to [s] save and exit or [c] continue or [e] exit without saving?"
    )
    choice = input().lower()
    if choice == "s":
        show_info("Saving and Exiting...")
        save_checkpoint(delta)
        os._exit(0)
    elif choice == "c":
        show_info("Continuing...")
        return
    elif choice == "e":
        show_info("Exiting without Saving...")
        os._exit(0)
    else:
        signal_handler(sig, frame, delta)
