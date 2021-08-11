import numpy as np
i32_nbytes = np.dtype(np.int32).itemsize

class KRXFile(object):
    def __init__(self, fname):
        self.fname = fname
        self.header_dtype = self.get_header_dtype() # todo use py38+ @cached_property
        self.read_header()

    def __repr__(self):
        return f"KRXFile({self.fname})"
        
    def get_header_dtype(self):
        dt_test = np.fromfile(self.fname, dtype=np.int32, count=2)
        if np.all(dt_test != 0):
            return np.int32
        else:
            return np.int64
    
    def read_header(self):
        with open(self.fname, 'rb') as f:
            hdr0 = np.fromfile(f, dtype=self.header_dtype, count=1)[0]
            hdr1 = np.fromfile(f, dtype=self.header_dtype, count=hdr0)
            hdr2 = np.fromfile(f, dtype=self.header_dtype, count=2)

            if hdr2[0] != 0:
                # DimSize+Len+MSA MapSizeArray
                map_size_arr = np.fromfile(f, dtype=self.header_dtype, count=hdr2[1])
                no_y = map_size_arr[0]
                no_e = map_size_arr[1]
                self.map_size = map_size_arr[2:]
            else:
                # old KRX without dimension info MapSizeArray
                no_y = hdr1[1]
                no_e = hdr1[2]
                self.map_size = np.array([len(hdr1)//3])
        
        assert np.prod(self.map_size) == len(hdr1)//3
        hdr1 = hdr1.reshape(-1, 3)
        self.page_start = hdr1[:, 0]
        self.page_shape = hdr1[:, 1:]

    @property
    def num_pages(self):
        return np.prod(self.map_size)
    
    def page(self, n=0):
        assert 0 <= n < self.num_pages
        return np.memmap(self.fname, mode='r', 
                         dtype=np.int32, offset=self.page_start[n]*i32_nbytes, 
                         shape=tuple(self.page_shape[n, ::-1]), order='F')

    def page_metadata(self, n=0):
        assert 0 <= n < self.num_pages
        page_size = self.page_start[n] + np.prod(self.page_shape[n])
        with open(self.fname, 'rb') as f:
            f.seek(page_size*i32_nbytes)
            hdr_len = np.fromfile(f, dtype=np.int32, count=1)[0]
            return f.read(hdr_len).decode('utf8')
    
    def export_page_txt(self, out_fname, n=0):
        with open(out_fname, 'w', newline='') as f:
                f.write(self.page_metadata(n))
                np.savetxt(f, self.page(n), delimiter='\t', fmt="%d")
