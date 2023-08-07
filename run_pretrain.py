import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.Generator import Generator
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from model.GANInstructor import GANInstructor
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params, parse_eval_args
from utilities.run_model import train_epoch, eval_model


def main():
	"""
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

	instructor = GANInstructor()
	#instructor.preTrainDiscriminator()
	instructor.trainBoth()



if __name__ == "__main__":
	main()
