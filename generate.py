import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.Generator import Generator
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam
from utilities.constants import *
from utilities.device import get_device, use_cuda


def randomiseTempo(generated):
    for idx, value in enumerate(generated):
        if value >= 256 and value < 356:
            #							never below 256			never above 356
            generated[idx] = random.randint(max(256,generated[idx]-10), min(generated[idx]+10,356))
            #generated[idx] = random.randint(256,356)
    return generated


def randomiseVolume(generated):
    for idx, value in enumerate(generated):
        if value >= 356:
            #							never below 356			never above 389
            generated[idx] = random.randint(max(356,generated[idx]-10), min(generated[idx]+10,389))
    return generated

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = encode_midi(f)
        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = Generator(n_layers=args.n_layers, num_heads=args.num_heads,
                      d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                      max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)
    print("primer DONE!")

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.sample(primer[:args.num_prime], args.target_seq_length)
            f_path = os.path.join(args.output_dir, "beam.mid")
            decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            rand_seq = model.sample(primer[:args.num_prime], args.target_seq_length)
            #rand_seq = randomiseVolume(rand_seq)
            f_path = os.path.join(args.output_dir, "rand.mid")
            decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)




if __name__ == "__main__":
    main()
