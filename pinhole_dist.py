import numpy as np
from astropy.table import Table


def calc_dist_close(avelis ='lis.lis', calc_err=True):
    '''
    '''

    r = []
    x = []
    y = []
    r_err = []
    avetab = Table.read(avelis, format='ascii.no_header')
    for f in  avetab['col1']:
        _tab = Table.read(f, format='ascii.fixed_width')
        for i in range(len(_tab)):
            _r = np.sqrt((_tab['x']-_tab['x'][i])**2 + (_tab['y'] -_tab['y'][i])**2)
            _pbool = _r < 550
            _pbool = (100 > np.abs(_tab['x']-_tab['x'][i])) + (100 > np.abs(_tab['y']-_tab['y'][i]))
            for kk in range(len(_pbool)):
                if _pbool[kk] and kk != i:

                    if _r[kk] < 550:
                        r.append(_r[kk])
                    elif _r[kk] > 550 and _r[kk] < 1100:
                        r.append(_r[kk]/2.0)
                    elif _r[kk] > 1100 and _r[kk] < 1650:
                        r.append(_r[kk]/3.0)
                    elif _r[kk] > 1650  and _r[kk] < 2100:
                        r.append(_r[kk]/4.0)
                    x.append(_tab['x'][kk])
                    y.append(_tab['y'][kk])
                    if calc_err:
                        #assume err value of 66 nm, and 2.2 micron pixels
                        #import pdb;pdb.set_trace()
                        err_val = 66.0 / 2200.0
                        drp = np.sqrt((_tab['x'][kk]-_tab['x'][i] -err_val)**2 + (_tab['y'][kk] -_tab['y'][i] - err_val )**2)
                        drm = np.sqrt((_tab['x'][kk]-_tab['x'][i] +err_val)**2 + (_tab['y'][kk] -_tab['y'][i] + err_val )**2)
                        _err = np.abs(_r[kk] - drp) + np.abs(_r[kk] -drm)
                        r_err.append(_err)

    if not calc_err:
        return r, x, y
    else:
        return r, x, y, r_err
