import io
import struct
import numpy as np

from .io import load


class KRXFile(object):
    def __init__(self, fname, zip_fname=None):
        self.fname = fname
        self.zip_fname = zip_fname
        self._read_header()

    def __repr__(self):
        return f"KRXFile({self.fname})"
    
    def _read_header(self):
        with load(self.fname, zip_fname=self.zip_fname, mode='rb') as f:
            endianness, char, itemsize = '<', 'i', 4
            dt_test = np.array(struct.unpack(endianness + 2*char, f.read(2 * itemsize)))
            if not all(dt_test != 0):  # 8byte int
                char, itemsize = 'q', 8
            f.seek(0)
            hdr0 = struct.unpack(endianness + char, f.read(itemsize))[0]
            hdr1 = struct.unpack(endianness + char * hdr0, f.read(itemsize * hdr0))
            hdr2 = struct.unpack(endianness + char * 2, f.read(itemsize * 2))

            if hdr2[0] != 0:
                # DimSize+Len+MSA MapSizeArray
                map_size_arr = struct.unpack(endianness + char * hdr2[1], f.read(itemsize * hdr2[1]))
                no_y = map_size_arr[0]
                no_e = map_size_arr[1]
                self.map_size = map_size_arr[2:]
            else:
                # old KRX without dimension info MapSizeArray
                no_y = hdr1[1]
                no_e = hdr1[2]
                self.map_size = np.array([hdr0//3])
        
        assert np.prod(self.map_size) == hdr0//3
        hdr1 = np.array(hdr1).reshape(-1, 3)
        self.page_start = hdr1[:, 0]
        self.page_shape = hdr1[:, 1:]

    @property
    def num_pages(self):
        return np.prod(self.map_size)
    
    def page(self, n=0):
        dt = np.dtype('int32')
        assert 0 <= n < self.num_pages
        with load(self.fname, zip_fname=self.zip_fname, mode='rb') as f:
            if isinstance(f, io.BufferedReader):
                # mmap normal files
                return np.memmap(f, mode='r', dtype=dt,
                                 offset=self.page_start[n] * dt.itemsize,
                                 shape=tuple(self.page_shape[n, ::-1]), order='F')
            else:
                # for zipped, gzipped, files etc. load data into memory
                f.seek(self.page_start[n] * dt.itemsize)
                nbytes = dt.itemsize * np.prod(self.page_shape[n, ::-1])
                return np.frombuffer(f.read(nbytes), dtype=dt).reshape(tuple(self.page_shape[n])).T

    def page_metadata(self, n=0):
        dt = np.dtype('int32')
        assert 0 <= n < self.num_pages
        page_size = self.page_start[n] + np.prod(self.page_shape[n])
        with load(self.fname, zip_fname=self.zip_fname, mode='rb') as f:
            f.seek(page_size * dt.itemsize)
            hdr_len = struct.unpack('<' + dt.char, f.read(dt.itemsize))[0]
            return f.read(hdr_len).decode('utf8')
    
    def export_page_txt(self, out_fname, n=0):
        with open(out_fname, 'w', newline='') as f:
            f.write(self.page_metadata(n))
            np.savetxt(f, self.page(n), delimiter='\t', fmt="%d")
