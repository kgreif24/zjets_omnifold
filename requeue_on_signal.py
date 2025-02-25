""" requeue_on_signal.py - This program propagates the SIGUSR1 and
SIGTERM signals to a process given as a command line argument.
This will be the PID of the "run_omnifold.py" script.
It also handles requeueing of the job in the event of a timeout
or preemption.

Author: Kevin Greif
Date: 12-13-2024
"""

import os
import time
import signal
import argparse


parser = argparse.ArgumentParser(description="Propagate signal to a process.")
parser.add_argument("--pid", type=int, help="Process ID to send signal to.")
args = parser.parse_args()


def handle_signal(signum, frame):
    """ Signal handler for the program. """
    print(f"Received signal {signum}, propagating to {args.pid}.")
    os.kill(args.pid, signum)
    time.sleep(60)
    os.system("sync")
    # Only need to requeue on timeout, not preemption
    if signum == signal.SIGUSR1:
        print("Wait 20 second for requeue")
        time.sleep(120)
        print("Requeue job! This also terminates the current job")
        os.system("scontrol requeue $SLURM_JOB_ID")


signal.signal(signal.SIGUSR1, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Wait for signals to arrive
print(f"Listening for signals on PID {os.getpid()}.")
signal.pause()

# After we get signal, wait for a bit so signals send properly
time.sleep(30)
