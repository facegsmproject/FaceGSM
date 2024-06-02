import tensorflow as tf
import sys
from utils.error_handling import show_info

def save_checkpoint(deltaUpdated):
    show_info("Saving Checkpoint...")
    checkpoint = tf.train.Checkpoint(delta=deltaUpdated)
    manager = tf.train.CheckpointManager(
        checkpoint, "./checkpoints", max_to_keep=3
    )
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
    print("\nDo you want to [s] save and exit or [c] continue?")
    choice = input().lower()
    if choice == 's':
        show_info("Saving and Exiting...")
        save_checkpoint(delta)
        sys.exit()
    elif choice == 'c':
        show_info("Continuing...")
        return
    else:
        signal_handler(sig, frame, delta)
