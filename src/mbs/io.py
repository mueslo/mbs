import os
import re
import io
import numpy as np
import datetime
import zipfile
import gzip
import warnings
from contextlib import contextmanager
from collections import OrderedDict
from collections.abc import Iterable

mbs_timestamp = lambda s: datetime.datetime.strptime(s.strip(), "%d/%m/%Y   %H:%M")
mbs_timestamp2 = lambda s: datetime.datetime.strptime(s.strip(), "%m/%d/%Y   %I:%M %p")
info_timestamp = lambda s: datetime.datetime.strptime(s.strip(), "%d/%m/%Y %H:%M:%S")
frame_unit = datetime.timedelta(seconds=0.001)
fname_re = re.compile(r'.*(?P<number>\d{5})_(?P<region>\d{5}).txt')
info_re = re.compile(r"^([^:]+):\s*([^(]+)\s*(\(([^)]+)\))?")


def is_mbs_filename(path):
     fname = os.path.basename(path)
     if fname_re.fullmatch(fname):
         return True
     return False


def mbs_boolean(s):
    if s.lower() == 'yes':
        return True
    elif s.lower() == 'no':
        return False
    raise ValueError


@contextmanager
def load(fname, zip_fname=None, mode='rt'):
    """Decorator that opens regular files, gzipped files and
    files contained within zip folders (e.g. archived measurements)"""
    if zip_fname is None:
        if fname.endswith('.gz'):
            open_func = gzip.open
        else:
            open_func = open
        with open_func(fname, mode) as f:
            yield f
    else:
        with zipfile.ZipFile(zip_fname) as zip_f:
            try:
                fname = zip_f.getinfo(fname)
            except KeyError:
                raise IOError(f'{zip_fname} does not contain {fname}')
            with zip_f.open(fname, ''.join(m for m in mode if m in ('r', 'w'))) as f:
                if 'b' in mode:
                    yield f
                else:  # r -> rt as normal open()
                    with io.TextIOWrapper(f) as f:
                        yield f


# from https://stackoverflow.com/a/2437645/
class LimitedSizeDict(OrderedDict):
    """Measurement cache"""
    # todo: separate caches and sizes for metadata and data
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
                key = self.popitem(last=False)[0]
                print(f'INFO: Retiring spectrum <{key}> from IO cache')


def parse_lines(lines, metadata_only=False):
        data_flag = False
        data = []
        metadata = OrderedDict()
        for line in lines:
            if data_flag:
                data.append(list(map(float, line.split())))
            elif line.startswith('DATA:'):
                if metadata_only:
                    return metadata
                data_flag = True
            else:
                # fix for old files
                if line.startswith('TIMESTAMP:') and not line.startswith('TIMESTAMP:\t'):
                    line = line.replace('TIMESTAMP:', 'TIMESTAMP:\t', 1)

                name, val = line.split('\t', 1)
                val = val.strip()
                if not name and not val:
                    continue
                for T in (int, float, mbs_timestamp, mbs_timestamp2, mbs_boolean):
                    try:
                        val = T(val)
                        break
                    except Exception as e:
                        continue

                if name in metadata:
                    warnings.warn(
                        f'Duplicate field {name} in metadata, '
                        f'overwriting previous value {metadata[name]} with {val}')
                metadata[name] = val
        if metadata['NoS'] != len(data[0]):
            assert metadata['NoS'] == len(data[0]) - 1 or len(data[0]) == 2  # resolved or integrated mode
            e_scale = np.linspace(metadata["Start K.E."], metadata["End K.E."]-metadata['Step Size'], len(data))
            assert np.allclose(e_scale, np.array(data)[:, 0])
            return np.array(data, dtype='uint32')[:, 1:], metadata

        return np.array(data, dtype='uint32'), metadata


io_cache = LimitedSizeDict(size_limit=128)


def parse_data(fname, metadata_only=False, zip_fname=None):
    try:
        key = (fname, metadata_only, zip_fname)
        file = zip_fname or fname
        mtime = os.path.getmtime(file)
        rv, mtime_cached = io_cache[key]

        if mtime != mtime_cached:
            print('File {} changed on disk, reloading...'.format(file))
            raise KeyError

        return rv

    except KeyError as ke:
        with load(fname, zip_fname) as f:
            rv = parse_lines(f, metadata_only=metadata_only)
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


class MBSFilePathGenerator(object):
    def __init__(self, prefix, directory=None, zip_fname=None):
        self.prefix = prefix
        self.directory = directory or ""
        self.zip_fname = zip_fname

    def __call__(self, number, region=None):
        if isinstance(number, Iterable):
            return [self(n, region) for n in number]
        if isinstance(region, Iterable):
            return [self(number, r) for r in region]
        
        if region is None:
            num_re = re.compile(r'{}{:05d}_\d{{5}}.txt'.format(self.prefix, number))
            paths = list(filter(lambda x: num_re.fullmatch(x), os.listdir(self.directory or '.')))
            if self.directory:
                paths = [os.path.join(self.directory, p) for p in paths]
            if not paths:
                raise Exception('No files found for {}{:05d}_#####.txt'.format(self.prefix, number))
            elif len(paths) == 1:
                return paths[0]
            return paths

        fname = "{}{:05d}_{:05d}.txt".format(self.prefix, number, region)
        return os.path.join(self.directory, fname)
