import torch
import numpy as np
import torch.nn.functional as functional
from src.sparse_torch.csx_matrix import CSRMatrix3d, CSCMatrix3d
import torch_geometric as pyg
import sys
from easydict import EasyDict as edict
from torch.nn import DataParallel
from time import time

import argparse
from src.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=16, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch, 'VISUAL.EPOCH', args.epoch])

    assert len(cfg.MODULE) != 0, 'Please specify a module name in your yaml file (e.g. MODULE: models.PCA.model).'
    assert len(cfg.DATASET_FULL_NAME) != 0, 'Please specify the full name of dataset in your yaml file (e.g. DATASET_FULL_NAME: PascalVOC).'

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args

def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)


class Timer:
    def __init__(self):
        self.t = time()
        self.tk = False

    def tick(self):
        self.t = time()
        self.tk = True

    def toc(self, tick_again=False):
        if not self.tk:
            raise RuntimeError('not ticked yet!')
        self.tk = False
        before_t = self.t
        cur_t = time()
        if tick_again:
            self.t = cur_t
            self.tk = True
        return cur_t - before_t

def load_model(model, path, strict=True):
    if isinstance(model, DataParallel):
        module = model.module
    else:
        module = model
    missing_keys, unexpected_keys = module.load_state_dict(torch.load(path), strict=strict)
    if len(unexpected_keys) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in missing_keys)))
        
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(indent_cnt=0)
def print_easydict(inp_dict: edict):
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            print('{}{}:'.format(' ' * 2 * print_easydict.indent_cnt, key))
            print_easydict.indent_cnt += 1
            print_easydict(value)
            print_easydict.indent_cnt -= 1

        else:
            print('{}{}: {}'.format(' ' * 2 * print_easydict.indent_cnt, key, value))

@static_vars(indent_cnt=0)
def print_easydict_str(inp_dict: edict):
    ret_str = ''
    for key, value in inp_dict.items():
        if type(value) is edict or type(value) is dict:
            ret_str += '{}{}:\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key)
            print_easydict_str.indent_cnt += 1
            ret_str += print_easydict_str(value)
            print_easydict_str.indent_cnt -= 1

        else:
            ret_str += '{}{}: {}\n'.format(' ' * 2 * print_easydict_str.indent_cnt, key, value)

    return ret_str



def count_parameters(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)



def pad_tensor(inp):
    """
    Pad a list of input tensors into a list of tensors with same dimension
    :param inp: input tensor list
    :return: output tensor list
    """
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(functional.pad(t, pad_pattern, 'constant', 0))

    return padded_ts





def data_to_cuda(inputs):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    #print(type(inputs))
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor, CSRMatrix3d, CSCMatrix3d]:
        inputs = inputs.cuda()
    elif type(inputs) in [pyg.data.Data, pyg.data.Batch]:
        inputs = inputs.to('cuda')
    else:
        inputs = inputs.to('cuda')
    return inputs




class LogWriter(object):
    def __init__(self, stdout, path, mode):
        self.path = path
        self._content = ''
        self._stdout = stdout
        self._file = open(path, mode)

    def write(self, msg):
        while '\n' in msg:
            pos = msg.find('\n')
            self._content += msg[:pos + 1]
            self.flush()
            msg = msg[pos + 1:]
        self._content += msg
        if len(self._content) > 1000:
            self.flush()

    def flush(self):
        self._stdout.write(self._content)
        self._stdout.flush()
        self._file.write(self._content)
        self._file.flush()
        self._content = ''

    def __del__(self):
        self._file.close()


class Logger(object):
    def __init__(self, path, mode='w+'):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = LogWriter(self._stdout, self.path, self.mode)
        sys.stdout = self._file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout