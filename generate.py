from base64 import encode
import torch
from utils.audio import save_wav
import argparse
import os
import sys
import time
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator
from utils.util import mu_law_encode, mu_law_decode
from utils.bert import bert_and_token

def attempt_to_restore(generator, discriminator ,checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def create_model(args):

    generator = Generator(args.local_condition_dim, args.z_dim)
    discriminator = Discriminator()

    return generator, discriminator

def get_vec_from_numpy(path, bert, tokenizer):
    vec = np.load(path)
    vec = vec[1:-1, :]
    return torch.FloatTensor(vec).unsqueeze(0).transpose(1, 2)

def get_vec_from_model(sentence, bert_model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = bert_model(**inputs)
    vec = outputs.last_hidden_state.detach()[:, 1:-1, :].transpose(1, 2)

def synthesis(args):

    bert_model, tokenizer = bert_and_token()

    generator, discriminator = create_model(args)
    if args.resume is not None:
       attempt_to_restore(generator, discriminator, args.resume, args.use_cuda)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    generator.to(device)
    discriminator.to(device)

    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)
    dir_list = os.listdir(os.path.join(args.input, 'audio'))
    dir_list = dir_list[0:1]
    print(dir_list)
    avg_rtf = []
    for filename in dir_list:
        start = time.time()

        style_sample = np.load(os.path.join(args.input, 'audio', filename))
        style_sample = style_sample[0:67500]
        style_input = torch.FloatTensor(style_sample).unsqueeze(0).unsqueeze(0)

        vec = get_vec_from_numpy(os.path.join(args.input, 'vec', filename), bert_model, tokenizer)[:, :, 0:5]
        z = discriminator(style_input, encode=True)

        audios = generator(vec, z)
        audios = audios.cpu().squeeze().detach().numpy()

        name = filename.split('.')[0]

        style_sample = mu_law_decode(mu_law_encode(style_sample))
        save_wav(np.squeeze(style_sample), '{}/{}_target.wav'.format(output_dir, name))
        save_wav(np.asarray(audios), '{}/{}_style.wav'.format(output_dir, name))
        time_used = time.time() - start
        rtf = time_used / (len(audios) / 24000)
        avg_rtf.append(rtf)
        print("Time used: {:.3f}, RTF: {:.4f}".format(time_used, rtf))

    print("Average RTF: {:.3f}".format(sum(avg_rtf) / len(avg_rtf)))

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]


    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of tests data')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default="logdir")
    parser.add_argument('--local_condition_dim', type=int, default=768)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=False)
    sys.argv = ['generate.py']
    args = parser.parse_args()
    synthesis(args)

if __name__ == "__main__":
    main()
