import numpy as np
from astropy.table import Table 
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import pidly
from astropy.io import fits





def read_psf(infile, psf_size=32, ds=18):
    '''
    '''

    tab = Table.read(infile, format='ascii.no_header', data_start=ds)
    psf_ar = np.zeros((psf_size, psf_size))
    for i in range(len(tab)):
        _dat =tab[i].as_void() 
        for j in range(len(tab[i])):
            psf_ar[i,j] = _dat[j]
        #import pdb;pdb.set_trace()
    return psf_ar

def read_pin_im(infile, ds=10):
    tab = Table.read(infile, format='ascii.no_header', data_start=ds)
    #assume image is spare
    im_size = len(tab)
    im_ar = np.zeros((im_size, im_size))
    for i in range(len(tab)):
        _dat = tab[i].as_void()
        for j in range(len(tab[i])):
            im_ar[i,j] = _dat[j]
    return im_ar
def plot_psfs(fig_n=1):

    infiles = []
    psfs = []
    for i in range(9):
        _psf = read_psf('IF'+str(i+1)+'.TXT')
        psfs.append(_psf)

    plt.figure(fig_n)
    plt.clf()
    plt.gray()
    cc = center_of_mass(psfs[4])
    for i in range(len(psfs)):
        plt.subplot(3,3,i+1)
        plt.imshow(psfs[i])
        _c = center_of_mass(psfs[i])
        plt.title('x '+str(round((_c[0])*3.706*1000))+ ' nm y '+str(round((_c[1])*3.706*1000))+' nm')
        plt.colorbar()

    

def write_fits(inlis, pad_pix=40):
    '''
    writes fits files from zemax text files
    pad_pix is the width of zeros added to the image
    '''

    intab = Table.read(inlis, format='ascii.no_header')

    for ff in intab['col1']:
        _im = read_pin_im(ff, ds=16)
        fits.writeto('psf_'+ff.replace('.txt', '.fits').replace('.TXT', '.fits'), _im) 
        _outim = np.pad(_im, pad_pix, 'constant')
        fits.writeto(ff.replace('.txt', '.fits').replace('.TXT', '.fits'), _outim) 
    

def cross_cor(inlis):
    '''
    '''
    intab = Table.read(inlis, format='ascii.no_header')
    
    for ff in intab['col1']:
        imstr = ff.replace('.txt', '.fits').replace('.TXT', '.fits')
        x = []
        y = []
        c = []
        for gg in intab['col1']:
            psfstr = 'psf_'+gg.replace('.txt', '.fits').replace('.TXT', '.fits')
            _im = fits.open(imstr)
            xr = (1.0*_im[0].data.shape[0])/2.0
            yr = (1.0*_im[0].data.shape[1])/2.0
            _x, _y, _f, _c = find(imstr, psfstr, xr, yr)
            x.append(_x)
            y.append(_y)
            c.append(_c)
        _outtab = Table()
        _outtab['x'] = x
        _outtab['y'] = y
        _outtab['c'] = c
        _outtab.write('stf_'+ff, format='ascii.basic')

def correct_txt(inlis):
    '''
    For each input PSF
    '''
    x = []
    y = []
    intab = Table.read(inlis, format='ascii.no_header')
    for ff in intab['col1']:
        _tab = Table.read(ff, format='ascii.basic')
        x.append(_tab['x'].data)
        y.append(_tab['y'].data)
    x = np.array(x)
    y = np.array(y)
    print(np.std(x, axis=1), np.std(y, axis=1))
    
    #get mean of each "star"
    xm = np.mean(x, axis=1)
    ym = np.mean(y, axis=1)
    dx = []
    dy = []
    
    for i in range(x.shape[1]):
        #subtract a given measurment of the star from the mean, take average of these offsets
        _dx = np.mean(x[:,i] - xm)
        _dy = np.mean(y[:,i] - ym)
        dx.append(_dx)
        dy.append(_dy)

    dx = np.array(dx)
    dy = np.array(dy)
    #import pdb;pdb.set_trace()
    #now add in offsets
    for i in range(len(dx)):
        x[:,i] = x[:,i] - dx[i]
        y[:,i] = y[:,i] - dy[i]
    xerr = np.std(x, axis=1)
    yerr = np.std(y, axis=1)

    summ = Table(data=[xerr, yerr], names=['xerr', 'yerr'])
    summ.write('summary.txt', format='ascii.basic')
    
    for i, ff in enumerate(intab['col1']):
        _out = Table()
        _out['x'] = x[i,:]
        _out['y'] = y[i,:]
        _out.write('cor_'+ff, format='ascii.basic')

    print xerr, yerr
        
        
def find(imfile,psffile,  xr, yr, abs_thresh=.1, corr=0.5):
    '''
    uses a known reference psf to find the positions for a star in the new frame
    '''

    idl = pidly.IDL()
    idl.xr = np.array(xr)
    idl.yr = np.array(yr)
    idl('corr = '+str(corr))
    idl("fits_read, '"+imfile+"', image, hdr")
    idl("fits_read, '"+psffile+"', psf, hdr")
    idl('starfinder, image, psf, background = background,'+str(abs_thresh)+',corr, N_ITER = 2, x, y, f, sx, sy, sf, c,STARS = stars')
    return idl.x, idl.y, idl.f, idl.c


def plot_psf_with_err(inlis, err_f='summary.txt',com_diff='com_diff.txt',  pix_scale=3.705, t=[30,28,26,24], ximloc=[-31.2, -10.6, 10.0], yimloc=[9.41, -9.7,  -28.5]):
    '''
    '''

    intab = Table.read(inlis, format='ascii.no_header')
    errors = Table.read(err_f, format='ascii.basic')
    com_diff = Table.read(com_diff, format='ascii.basic')

    coo = np.meshgrid(ximloc, yimloc)

    plt.clf()
    plt.figure(1)
    plt.gray()
    for i in range(len(intab)):
        _im = fits.getdata(intab['col1'][i])
        plt.subplot(3, 3, i+1)
        plt.imshow(_im, vmin=0, vmax=0.65, origin='lower')
        plt.title('IMA '+str(coo[0].flatten()[i]) + ' , '+str(coo[1].flatten()[i]) +' mm')
        plt.text(1, t[0], 'x rms '+str(np.round(errors['xerr'][i]*pix_scale * 1000.0) )+' nm', color='white')
        plt.text(1, t[1], 'y rms '+str(np.round(errors['yerr'][i]*pix_scale * 1000.0) )+' nm', color='white')
        plt.text(1,t[2], 'com x '+str(np.round(com_diff['x'][i]*pix_scale*1000.0))+' nm', color='white')
        plt.text(1,t[3], 'com y '+str(np.round(com_diff['y'][i]*pix_scale*1000.0))+' nm', color='white')

    plt.savefig('psf_sum.png')
    plt.show()
    

def compare2com(inlis, psf_ref=4):
    intab = Table.read(inlis, format='ascii.no_header')
    psf_im = fits.getdata(intab['col1'][psf_ref])
    coo = Table.read('stf_'+intab['col1'][psf_ref].replace('.fits', '.txt').replace('psf_', ''), format='ascii.basic')
    com = center_of_mass(psf_im)
    dx = com[1] - coo['x'][psf_ref]
    dy = com[0] - coo['y'][psf_ref]
    import pdb;pdb.set_trace()
    xc = []
    yc =[]
    xs = []
    ys = []
    for i in range(len(intab)):
        _im = fits.getdata(intab['col1'][i])
        _com = center_of_mass(_im)
        _coo = Table.read('stf_'+intab['col1'][i].replace('.fits', '.txt').replace('psf_', ''), format='ascii.basic')

        xs.append(_coo['x'][psf_ref])
        ys.append(_coo['y'][psf_ref])
        xc.append(_com[1])
        yc.append(_com[0])

    xs = np.array(xs)
    ys = np.array(ys)
    xc = np.array(xc)
    yc = np.array(yc)
    dx = np.mean(xs - xc)
    dy = np.mean(ys - yc)
    diffx = xs - xc - dx
    diffy = ys - yc - dy
    outab = Table(data=[diffx, diffy], names=['x','y'])
    outab.write('com_diff.txt', format='ascii.basic')
