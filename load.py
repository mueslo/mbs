import os
import re
import io
import numpy as np
import datetime
import zipfile
from contextlib import contextmanager
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache

mbs_timestamp = lambda s: datetime.datetime.strptime(s, "%d/%m/%Y   %H:%M")
frame_unit = datetime.timedelta(seconds=0.001)
fname_re = re.compile(r'.*(?P<number>\d{5})_(?P<region>\d{5}).txt')



def is_mbs_filename(path):     
     fname = os.path.basename(path)
     if fname_re.fullmatch(fname):
         return True
     return False


def stats(metadata):
    s = {}
    if metadata['AcqMode'] == 'Fixed':
        s['acqtime'] = metadata['ActScans'] * metadata['Frames Per Step'] * frame_unit
        s['eff_acqtime'] = s['acqtime']
    elif metadata['AcqMode'] == 'Swept':
        s['acqtime'] = metadata['ActScans'] * metadata['TotSteps'] * metadata['Frames Per Step'] * frame_unit
        s['eff_acqtime'] =  metadata['ActScans'] * metadata['No. Steps'] * metadata['Frames Per Step'] * frame_unit
    elif metadata['AcqMode'] == 'Dither':
        print('dither!')
        s['acqtime'] = metadata['ActScans'] * metadata['TotSteps'] * metadata['Frames Per Step'] * frame_unit
        s['eff_acqtime'] = metadata['ActScans'] * metadata['TotSteps'] * (metadata['No. Steps'] - metadata['TotSteps'])/metadata['No. Steps'] * metadata['Frames Per Step'] * frame_unit
        
    s['duration'] = metadata['TIMESTAMP:'] - metadata['STim']
    return s


@contextmanager
def load(fname, zip_fname=None):
    if zip_fname is None:
        with open(fname, 'r') as f:
            yield f
    else:
        with zipfile.ZipFile(zip_fname) as zip_f:
            with zip_f.open(fname, 'r') as f:
                with io.TextIOWrapper(f) as f:
                    yield f


@lru_cache(maxsize=100, typed=False)
def parse_data(fname, metadata_only=False, zip_fname=None):
    with load(fname, zip_fname) as f:
        data_flag = False
        data = []
        metadata = OrderedDict()
        for line in f:
            if data_flag:
                data.append(list(map(float, line.split())))
            elif line.startswith('DATA:'):
                if metadata_only:
                    return metadata
                data_flag = True
            else:
                name, val = line.split('\t', 1)
                val = val.strip()

                for T in (int, float, mbs_timestamp):
                    try:
                        val = T(val)
                        break
                    except Exception as e:
                        continue

                metadata[name] = val
        if metadata['NoS'] != len(data[0]):
            print('WARNING NoS={}, data.shape={},{}'.format(
                metadata['NoS'], len(data), len(data[0])))
        return np.array(data, dtype='uint32'), metadata


class MBSFilePathGenerator(object):
    def __init__(self, prefix, directory):
        self.prefix = prefix
        self.directory = directory
    
    def __call__(self, number, region):
        if isinstance(number, Iterable):
            return [self(n, region) for n in number]
        if isinstance(region, Iterable):
            return [self(number, r) for r in region]
        
        fname = "{}{:05d}_{:05d}.txt".format(self.prefix, number, region)
        return os.path.join(self.directory, fname)
