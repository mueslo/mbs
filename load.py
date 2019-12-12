import os
import re
import io
import numpy as np
import datetime
import zipfile
from contextlib import contextmanager
from collections import OrderedDict, namedtuple
from collections.abc import Iterable

mbs_timestamp = lambda s: datetime.datetime.strptime(s.strip(), "%d/%m/%Y   %H:%M")
info_timestamp = lambda s: datetime.datetime.strptime(s.strip(), "%d/%m/%Y %H:%M:%S")
frame_unit = datetime.timedelta(seconds=0.001)
fname_re = re.compile(r'.*(?P<number>\d{5})_(?P<region>\d{5}).txt')
info_re = re.compile(r"^([^:]+):\s*([^(]+)\s*(\(([^)]+)\))?")
XASpectrum = namedtuple('XASpectrum', ['e', 'i0', 'sample', 'ref', 'i0_v', 'sample_v', 'ref_v', 'gap'])

def is_mbs_filename(path):     
     fname = os.path.basename(path)
     if fname_re.fullmatch(fname):
         return True
     return False


def mbs_boolean(s):
    if s == 'Yes':
        return True
    elif s == 'No':
        return False
    raise ValueError


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


# from https://stackoverflow.com/a/2437645/
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


io_cache = LimitedSizeDict(size_limit=128)
def parse_data(fname, metadata_only=False, zip_fname=None):
    def parse_data_inner():
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

                    for T in (int, float, mbs_timestamp, mbs_boolean):
                        try:
                            val = T(val)
                            break
                        except Exception as e:
                            continue

                    if name in metadata:
                        print('Warning, duplicate field', name)
                    metadata[name] = val
            if metadata['NoS'] != len(data[0]):
                assert metadata['NoS'] == len(data[0]) - 1
                e_scale = np.linspace(metadata["Start K.E."], metadata["End K.E."]-metadata['Step Size'], len(data))
                assert np.allclose(e_scale, np.array(data)[:, 0])
                return np.array(data, dtype='uint32')[:, 1:], metadata

            return np.array(data, dtype='uint32'), metadata

    try:
        key = (fname, metadata_only, zip_fname)
        file = fname or zip_fname
        mtime = os.path.getmtime(file)
        rv, mtime_cached = io_cache[key]

        if mtime != mtime_cached:
            print('File {} changed on disk, reloading...'.format(file))
            raise KeyError

        return rv

    except KeyError as ke:
        rv = parse_data_inner()
        io_cache[key] = (rv, mtime)
        return rv

def parse_info(fname, zip_fname=None):
    with load(fname, zip_fname) as f:
        info = OrderedDict()
        for line in f:
            line = info_re.match(line)
            quantity, value, unit = line.group(1, 2, 4)

            for T in [int, float, info_timestamp]:
                try:
                    value = T(value)
                    break
                except ValueError:
                    continue

            info[quantity] = (value, unit)
        return info


def parse_xas(fname, zip_fname=None):
    with load(fname, zip_fname) as f:
        arr = np.loadtxt(f)
        return XASpectrum(arr[:, 0],
                          arr[:, 5],
                          arr[:, 6],
                          arr[:, 4],
                          arr[:, 8],
                          arr[:, 9],
                          arr[:, 7],
                          arr[:, 2])


class MBSFilePathGenerator(object):
    def __init__(self, prefix, directory=None):
        self.prefix = prefix
        self.directory = directory or ""
    
    def __call__(self, number, region=None):
        if isinstance(number, Iterable):
            return [self(n, region) for n in number]
        if isinstance(region, Iterable):
            return [self(number, r) for r in region]
        
        if region is None:
            num_re = re.compile('{}{:05d}_\d{{5}}.txt'.format(self.prefix, number))
            paths = list(filter(lambda x: num_re.fullmatch(x), os.listdir(self.directory or '.')))
            if not paths:
                raise Exception('No files found for {}{:05d}_'.format(self.prefix, number))
            elif len(paths) == 1:
                return paths[0]
            return paths

        fname = "{}{:05d}_{:05d}.txt".format(self.prefix, number, region)
        return os.path.join(self.directory, fname)
