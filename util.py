from PIL import Image
import numpy as np
import radial_data
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from scipy.ndimage.measurements import  center_of_mass
from astropy.table import Table
from astropy.io import fits
import os, shutil


def conv_withIMAG(conlis, in_format='.pgm', batfile='conv.bat', out_format='.fits'):
    intab = Table.read(conlis, format='ascii.no_header')
    f = open(batfile, 'w')
    for i in range(len(intab)):
        f.write('convert '+intab['col1'][i].replace('*', '') +'  '+intab['col1'][i].replace('*', '').replace(in_format, out_format)+' \n')

    f.write('ls -1 *.fits > fits.lis \n')
    f.close()

def convlis_withIMAG(intab, in_format='.pgm', batfile='conv.bat', out_format='.fits'):
    f = open(batfile, 'w')
    for i in range(len(intab)):
        f.write('convert '+intab[i].replace('*', '') +'  '+intab[i].replace('*', '').replace(in_format, out_format)+' \n')

    f.write('ls -1 *.fits > fits.lis \n')
    f.close()

def ave_fits(fits_lis, outf='ave.fits'):
    intab = Table.read(fits_lis, format='ascii.no_header')

    dat = []
    nfiles = 244
    if len(intab) < nfiles:
        nfiles = len(intab)
    for i in range(nfiles):
        _tmp = fits.open(intab['col1'][i])
        dat.append(_tmp[0].data[:,:])
        _tmp.close()
        
    dat = np.array(dat)
    out = np.mean(dat, axis=0)
    fits.writeto(outf, out, clobber=True)
        
def avelis_fits(inlis, outf='ave.fits'):
    #intab = Table.read(fits_lis, format='ascii.no_header')

    dat = []
    for i in range(len(inlis)):
        _tmp = fits.open(inlis[i])
        dat.append(_tmp[0].data)
    dat = np.array(dat)
    out = np.mean(dat, axis=0)
    fits.writeto(outf, out)
    
def read_tif(imfile):
    im = Image.open(imfile)
    im = np.array(im)
    return im
def png2fits(bmplisF, pref=''):
    #assumes that bmp is 2D

    bmplis = Table.read(bmplisF, format='ascii.no_header')['col1']
    for imfile in bmplis:
        im = Image.open(imfile)
        im = np.array(im)
        data = im[:,:]
        #import pdb;pdb.set_trace()
        fits.writeto(pref+imfile.replace('.png','.fits'), data.astype('float'), clobber=True) 

def tif2fits(bmplisF, pref=''):
    #assumes that bmp is 2D

    bmplis = Table.read(bmplisF, format='ascii.no_header')['col1']
    for imfile in bmplis:
        im = Image.open(imfile)
        im = np.array(im)
        data = im#[:,:,0]
        #import pdb;pdb.set_trace()
        fits.writeto(pref+imfile.replace('.tif','.fits'), data.astype('float'), clobber=True) 
        
def bmp2fits(bmplisF, pref=''):
    #assumes that bmp is 2D

    bmplis = Table.read(bmplisF, format='ascii.no_header')['col1']
    for imfile in bmplis:
        im = Image.open(imfile)
        im.load()
        #data = np.reshape(np.array(im.im), (im.size[1], im.size[0]))
        data = np.reshape(np.array(im.im)[:,0], (im.size[1], im.size[0]))
        data  = data.astype('float32')
        #import pdb;pdb.set_trace()
        fits.writeto(pref+imfile.replace('.bmp','.fits'), data, clobber=True) 

def ppm2fits(ppm_lis, def_channel=2, pref=''):
    '''
    take text file of *.ppm images, and converts them into fits files
    def_channel is the default channel written into the fits file
    pref is a string that  an be appeneded to the front of the fits file names
    '''
    tab = Table.read(ppm_lis, format='ascii.no_header')
    #import pdb ;pdb.set_trace()
    for ff in tab['col1']:
        im = Image.open(ff)
        im = np.array(im)
        fits.writeto(pref+ff.replace('.ppm', '.fits'), im[:,:,def_channel], clobber=True)

def pgm2fits(pgm_lis, def_channel=2, pref=''):
    '''
    take text file of *.ppm images, and converts them into fits files
    def_channel is the default channel written into the fits file
    pref is a string that  an be appeneded to the front of the fits file names
    '''
    tab = Table.read(pgm_lis, format='ascii.no_header')
    #import pdb ;pdb.set_trace()
    for ff in tab['col1']:
        
        im = Image.open(ff)
        im.load()
        im = np.array(im)
        #import pdb;pdb.set_trace()
        fits.writeto(pref+ff.replace('.pgm', '.fits'), im, clobber=True)
    
    
def bin_rad(array, xcen, ycen, size, plot=False):
    '''
    '''

    cutout = array[xcen -size/2:xcen+size/2, ycen- size/2:ycen+size/2]
    rad = radial_data.radial_data(cutout)

    if plot:
        plt.figure(1)
        plt.clf()
        plt.scatter(rad.r, rad.mean)
        plt.show()

    return rad
     
def fit_gauss(array):

    g = models.Gaussian2D(x_stddev=1, y_stddev=1)
    fit_g = fitting.LevMarLSQFitter

    coo =  np.meshgrid(range(array.shape[0]), range(array.shape[1]))
    gmod = fit_g(g, coo[0], coo[1], array)

    return gmod


def back_sub(imar, sky_size=100):
    '''
    subtract out median of the sky_size X sky_size corner boxes
    return imar - sky_median
    '''

    imbool = np.ones(imar.shape, dtype='bool')
    imbool[sky_size:-sky_size, sky_size:-sky_size] = False
    med_sky = np.median(imar[imbool])
    return imar - med_sky, med_sky

def find_rad(tif_file, rad_size=1100, radfrac=0.25):
    '''
    finds the radius of bright circle in an image
    '''

    
    imar = read_tif(tif_file)[:,:,2]

    imar, bkg = back_sub(imar)

    com_coo= center_of_mass(imar)

    #print com_coo
    
    rad = bin_rad(imar, com_coo[0], com_coo[1], rad_size)

    peak_flux = np.mean(rad.mean[:5])
    #if radtype == 'half_peak':
    flux_cut = radfrac * peak_flux
    radius = np.max(rad.r[rad.mean > flux_cut])
    print 'radius type is ', radfrac
    print 'radius is ', radius
    print 'center of mass is ', com_coo
    print 'cut off flux is ', flux_cut
    print 'background is ', bkg
    
    return rad
    
def ave_group(im_f,g_size=3):
    '''
    averages images in groups of g_size
    writes average and rms scatter images for each
    '''
    imlis = Table.read(im_f, format='ascii.no_header')['col1']
    #read in all images
    imall = []
    for i in range(len(imlis)):
        imall.append(fits.open(imlis[i])[0].data)
    imall = np.array(imall)
        
    ibool = np.ones(len(imlis), dtype='bool')
   
    for i in range(len(imlis)):
        if ibool[i]:
            #average next three images
            _ave_im = np.mean(imall[i:i+3,:,:], axis=0)
            _std_im = np.std(imall[i:i+3,:,:], axis=0)
            fits.writeto('ave_'+str(i).zfill(3)+'.fits', _ave_im, clobber=True)
            fits.writeto('sig_'+str(i).zfill(3)+'.fits', _std_im, clobber=True)
            ibool[i:i+3] = False

    
            
        

def red_dir(dirname):

    os.chdir(dirname)
    dirs = os.listdir('.')

    for dd in dirs:
        os.chdir(dd)
        print os.getcwd()
        _dirs = os.listdir('.')
        #for _dd in _dirs:
        print 'working in directory ', dd
        all_files = os.listdir('.')
        pgm_lis = []
        for ff in all_files:
            if '.pgm' in ff:
                pgm_lis.append(ff)
                    #print ff
        #now convert the pgm files to fits files
        convlis_withIMAG(pgm_lis)
        os.system('sh conv.bat')
        #all_files = os.listdir('.')
        #fits_lis = []
        #for ff in all_files:
        #    if '.fits' in ff:
        #        fits_lis.append(ff)
        ave_fits('fits.lis', outf = '../'+dirname.replace('./','').replace('/','')+'_'+dd+'.fits')
        os.chdir('..')
        print 'finished averaging'
        #import pdb;pdb.set_trace()
        print os.getcwd()
        #os.chdir('..')

    os.chdir('..')
            
            
            
    
def fix_zfill(inlis):
    '''
    goes through list of images
    assumes that file numbes are sandwicheded between a  - and .
    takes each file number, fills with zeroes and saves a new file
    '''

    intab =  Table.read(inlis, format='ascii.no_header')
    for ff in intab['col1']:
        _ss = ff.split('-')
        _ssF = _ss[-1].split('.')
        _num = _ssF[0]
        _newnum = _num.zfill(5)
        newname = _ss[0]+_newnum+'.'+_ssF[-1]
        shutil.copy(ff, newname)
