import numpy as np
from astropy.io import fits
from scipy.ndimage.measurements import center_of_mass
from astropy.time import Time
from astropy.table import Table 
import numpy as np
def mkpostime(inlis='FIT.lis'):


    intab = Table.read(inlis, format='ascii.no_header')
    com = []
    for _ff in intab['col1']:
        _tmp = fits.open(_ff)
        _com  = center_of_mass(_tmp[0].data)
        com.append(_com)

    com = np.array(com)

    #grab approx time by linearly interpolating between first time and last time.  note, there are some errors in this approacj, but the times in the header are only good to 1s

    t0 = fits.getheader(intab['col1'][0])['DATE-OBS']
    tf = fits.getheader(intab['col1'][-1])['DATE-OBS']
    t0s = Time(t0).unix
    tfs = Time(tf).unix

    times = np.linspace(t0s, tfs, num=len(intab))
    return com, times
