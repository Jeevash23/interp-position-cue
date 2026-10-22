"""Serialization-only adapter for NumPy scalar booleans in frozen analysis outputs.

The original frozen analyses remain byte-for-byte unchanged. No numerical
computation, threshold, input, selection, or inference formula is replaced.
"""
import argparse,importlib
import numpy as np
import core as c


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,list):return [native(v) for v in value]
    if isinstance(value,tuple):return tuple(native(v) for v in value)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',choices=['analyze','replay_analysis'],required=True);p.add_argument('--run');p.add_argument('--output');a=p.parse_args()
    original=c.write
    def serialized(path,value):return original(path,native(value))
    c.write=serialized
    try:
        module=importlib.import_module(a.module)
        if a.module=='analyze':module.analyze(a.run,a.output)
        else:module.main()
    finally:c.write=original


if __name__=='__main__':main()
