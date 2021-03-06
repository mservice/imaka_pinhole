from astropy.table import Table
import numpy as np

def clean(inlis, ccut=0.85, xmax = 9633, ymax=8070, xmin=0, ymin=0, fmax = 50000, fmin= 17000):
    '''
    takes a 1 column list of starfinder files (inlis)
    trims the data per the keyword arguments and rights a timmed version of the file
    also adds a name column
    '''
    intab = Table.read(inlis, format='ascii.no_header')
    #coordiantes to block sources at
    xb = [6175]
    yb = [2707]
    
    
    for _ff in intab['col1']:
        _tab = Table.read(_ff, format='ascii.fixed_width')
        gg = (_tab['corr'] >  ccut) * (_tab['x'] < xmax) * (_tab['x'] > xmin) * (_tab['y'] < ymax) *(_tab['y'] > ymin) *(_tab['flux'] > fmin) *(_tab['flux'] < fmax)
        rb = np.ones(len(gg), dtype='bool')
        for jj in range(len(xb)):
            rr = (_tab['x'] - xb[jj])**2+(_tab['y'] - yb[jj])**2
            if np.min(rr) < 15:
                rb[np.argmin(rr)] = False
        gg = gg * rb
        ntab = _tab[gg]
        print(np.sum(gg))
        names  = [ 'star_'+str(x).zfill(5) for x in range(len(ntab))]
        ntab['name'] = names
        ntab['mag'] = -2.5*np.log10(ntab['flux'])
        ntab.write(_ff.replace('.', '_trim.'), format='ascii.fixed_width')
    
