from . import starfinder
from astropy.table import Table
from astropy.io import fits 
import numpy as np

def run_all_im(im_f, reffile,sig_file='sig.fits', bkg='none', find_ref=True, search=True):
    imlis = Table.read(im_f, format='ascii.no_header')['col1']
    for im in imlis:
        #import pdb;pdb.set_trace()
        if bkg == 'none':
            bkg = np.median(fits.open(im)[0].data)
            print bkg
        starfinder.run_stf(im, sig_file, bkg, reffile=reffile, find_ref=find_ref, search=search)
        
def run_all_ave(im_f, sig_f, reffile, bkg='none', find_ref=True, search=True):
    imlis = Table.read(im_f, format='ascii.no_header')['col1']
    siglis = Table.read(sig_f, format='ascii.no_header')['col1']
    for i,im in enumerate(imlis):
        #import pdb;pdb.set_trace()
        if bkg == 'none':
            bkg = np.median(fits.open(im)[0].data)
            print bkg
        starfinder.run_stf(im, siglis[i], bkg, reffile=reffile, find_ref=find_ref, search=search)



def run_ave(ave_f):
    avelis = Table.read(ave_f, format='ascii.no_header')['col1']
    
    for im in avelis:
        bkg = np.median(fits.open(im)[0].data)
        import pdb;pdb.set_trace()
        starfinder.run_stf(im, im.replace('ave','sig'), bkg, find_ref=False, search=False)
