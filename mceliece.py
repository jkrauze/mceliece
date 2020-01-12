#!/usr/bin/env python3
"""McEliece v0.1

Usage:
  mceliece.py [options] enc PUB_KEY_FILE [FILE]
  mceliece.py [options] dec PRIV_KEY_FILE [FILE]
  mceliece.py [options] gen M N T PRIV_KEY_FILE PUB_KEY_FILE
  mceliece.py (-h | --help)
  mceliece.py --version

Options:
  -b, --block        Interpret input/output as
                       block stream.
  -i, --poly-input   Interpret input as polynomial
                       represented by integer array.
  -o, --poly-output  Interpret output as polynomial
                       represented by integer array.
  -h, --help         Show this screen.
  --version          Show version.
  -d, --debug        Debug mode.
  -v, --verbose      Verbose mode.
"""
from docopt import docopt
from mceliece.mceliececipher import McElieceCipher
from sympy.abc import x
from sympy import ZZ, Poly
from padding.padding import *
import numpy as np
import sys
import logging
import math

log = logging.getLogger("mceliece")

debug = False
verbose = False


def generate(m, n, t, priv_key_file, pub_key_file):
    mceliece = McElieceCipher(m, n, t)
    mceliece.generate_random_keys()
    np.savez_compressed(priv_key_file, m=m, n=n, t=t, S=mceliece.S, S_inv=mceliece.S_inv, G=mceliece.G, H=mceliece.H,
                        P=mceliece.P, P_inv=mceliece.P_inv, g_poly=mceliece.g_poly, irr_poly=mceliece.irr_poly)
    log.info(f'Private key saved to {priv_key_file} file')
    np.savez_compressed(pub_key_file, m=m, n=n, t=t, Gp=mceliece.Gp)
    log.info(f'Public key saved to {pub_key_file} file')


def encrypt(pub_key_file, input_arr, block=False):
    pub_key = np.load(pub_key_file, allow_pickle=True)
    mceliece = McElieceCipher(int(pub_key['m']), int(pub_key['n']), int(pub_key['t']))
    mceliece.Gp = pub_key['Gp']

    if not block:
        if mceliece.Gp.shape[0] < len(input_arr):
            raise Exception(f"Input is too large for current N. Should be {mceliece.Gp.shape[0]}")
        output = mceliece.encrypt(input_arr).to_numpy()
    else:
        input_arr = padding_encode(input_arr, mceliece.Gp.shape[0])
        input_arr = input_arr.reshape((-1, mceliece.Gp.shape[0]))
        output = np.array([])
        block_count = input_arr.shape[0]
        for i, b in enumerate(input_arr, start=1):
            log.info("Processing block {} out of {}".format(i, block_count))
            next_output = mceliece.encrypt(b).to_numpy()
            if len(next_output) < mceliece.Gp.shape[1]:
                log.debug(f"Padding added in block {i}")
                next_output = np.pad(next_output, (0, mceliece.Gp.shape[1] - len(next_output)), 'constant')
            output = np.concatenate((output, next_output))

    return np.array(output).flatten()


def decrypt(priv_key_file, input_arr, block=False):
    priv_key = np.load(priv_key_file, allow_pickle=True)
    mceliece = McElieceCipher(int(priv_key['m']), int(priv_key['n']), int(priv_key['t']))
    mceliece.S = priv_key['S']
    mceliece.S_inv = priv_key['S_inv']
    mceliece.H = priv_key['H']
    mceliece.G = priv_key['G']
    mceliece.P = priv_key['P']
    mceliece.P_inv = priv_key['P_inv']
    mceliece.g_poly = priv_key['g_poly']
    mceliece.irr_poly = priv_key['irr_poly']

    if not block:
        if len(input_arr) < mceliece.H.shape[1]:
            input_arr = np.pad(input_arr, (0, mceliece.H.shape[1] - len(input_arr)), 'constant')
        return mceliece.decrypt(input_arr)

    if len(input_arr)%mceliece.H.shape[1] > 0:
        input_arr = np.pad(input_arr, (0, mceliece.H.shape[1] - len(input_arr)%mceliece.H.shape[1]), 'constant')
    input_arr = input_arr.reshape((-1, mceliece.H.shape[1]))
    output = np.array([])
    block_count = input_arr.shape[0]
    for i, b in enumerate(input_arr, start=1):
        log.info("Processing block {} out of {}".format(i, block_count))
        log.debug(f"msg:{b}")
        next_output = mceliece.decrypt(b)
        if len(next_output) < mceliece.G.shape[0]:
            log.debug(f"Padding added in block {i}")
            next_output = np.pad(next_output, (0, mceliece.H.shape[1] - len(next_output)), 'constant')
        output = np.concatenate((output, next_output))
    return padding_decode(output.flatten(), mceliece.G.shape[0])


if __name__ == '__main__':
    args = docopt(__doc__, version='McEliece v0.1')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    if args['--debug']:
        ch.setLevel(logging.DEBUG)
    elif args['--verbose']:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.WARN)
    root.addHandler(ch)

    log.debug(args)
    poly_input = bool(args['--poly-input'])
    poly_output = bool(args['--poly-output'])
    block = bool(args['--block'])
    input_arr, output = None, None
    if not args['gen']:
        if args['FILE'] is None or args['FILE'] == '-':
            input = sys.stdin.read() if poly_input else sys.stdin.buffer.read()
        else:
            with open(args['FILE'], 'rb') as file:
                input = file.read()
        log.info("---INPUT---")
        log.info(input)
        log.info("-----------")
        if poly_input:
            input_arr = np.array(eval(input))
        else:
            input_arr = np.unpackbits(np.frombuffer(input, dtype=np.uint8))
        input_arr = np.trim_zeros(input_arr, 'b')
        log.info("POLYNOMIAL DEGREE: {}".format(max(0, len(input_arr) - 1)))
        log.debug("BINARY: {}".format(input_arr))

    if args['gen']:
        generate(int(args['M']), int(args['N']), int(args['T']), args['PRIV_KEY_FILE'], args['PUB_KEY_FILE'])
    elif args['enc']:
        output = encrypt(args['PUB_KEY_FILE'], input_arr, block=block)
    elif args['dec']:
        output = decrypt(args['PRIV_KEY_FILE'], input_arr, block=block)

    if not args['gen']:
        if poly_output:
            print(list(output.astype(np.int)))
        else:
            sys.stdout.buffer.write(np.packbits(np.array(output).astype(np.int)).tobytes())
