from imaka_pinhole import fit_all, match_pin
import numpy as np
from astropy.table import Table
import glob
import shutil
import os
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.modeling import models, fitting                                                                                                   
from astropy.modeling.models import custom_model 

def fitind():
    lisf = glob.glob('Pinhole*/obj*_sub.lis')
    print(lisf)

    x0 = 3901.0300000000002
    y0 = 3144.25
    rcut = 2900

    for i in range(len(lisf)):
        _in = Table.read(lisf[i], format='ascii.fixed_width')
        rad = ((_in['x']-x0)**2+(_in['y']-y0)**2)**0.5
        rb = rad < rcut

        _tr = _in[rb]
        os.remove('tmp.lis')
        _tr.write('tmp.lis', format='ascii.fixed_width')
        _lookup = glob.glob('lookup*.fits')
        for _ff in _lookup:
            os.remove(_ff)
            
        fit_all.fit_dist_single('tmp.lis', ang=15, gspace=168*1.3, order=4, mklook=True, print_errors=False)

        shutil.move('lookup_x.fits', lisf[i].replace('.lis','_x.fits'))
        shutil.move('lookup_y.fits', lisf[i].replace('.lis','_y.fits'))
        shutil.move('Legpx.txt', lisf[i].replace('.lis','_legpx.txt'))
        shutil.move('Legpy.txt', lisf[i].replace('.lis','_legpy.txt'))


def mkdist(inlis='obj.lis'):

    _t = fit_all.fit_dist_single('average_coo.txt', retrans=True, ang=15, gspace=168*1.3, order=4)
    lisf = Table.read(inlis, format='ascii.no_header')
    #apply the distortion tranformation to each file, make new average
    nlisf = []
    for _ff in lisf['col1']:
        print('working on table ',_ff)
        _in = Table.read(_ff, format='ascii.fixed_width')
        xn, yn = _t.evaluate(_in['x'], _in['y'])
        _in['x']=xn
        _in['y']=yn
        nlisf.append(_ff.split('/')[-1].replace('.lis','_tmp.lis'))
        _in.write(_ff.split('/')[-1].replace('.lis','_tmp.lis'), format='ascii.fixed_width')

    _out = Table()
    _out['fn'] = nlisf
    _out.write('obj_d.lis', format='ascii.no_header')

    fit_all.mkave('obj_d.lis')
    #_tn = fit_dist_single('



def plot_trans(inlis='obj.lis', plot_sep=False):
    '''
    '''

    xall, yall, fall, dx, dy = match_pin.match_all(inlis, retoff=True)
    xave = np.mean(xall, axis=1)
    yave = np.mean(yall, axis=1)

    dev_x = (xall.T - xave).T
    dev_y = (yall.T - yave).T

    plt.figure(1)
    plt.clf()

    if not  plot_sep:
        colors = ['black', 'blue', 'green', 'purple', 'magenta', 'green']
        for i in range(xall.shape[1]):
            q = plt.quiver(xave, yave, dev_x[:,i], dev_y[:,i], color=colors[i], label='Frame '+str(i), scale=1)

                 
        plt.quiverkey(q, 6500, 1500, 2*0.0238, label='2 mas', color='red', coordinates='data')
        plt.xlim(0,7000)
        plt.ylim(0,6500)
        plt.axes().set_aspect('equal')
        plt.ylabel("Y (pixels)")
        plt.xlabel("x (pixels)")
        plt.legend(loc='upper left')

    else:
        for i in range(xall.shape[1]):
            plt.figure(i)
            plt.clf()

            plt.title('Frame '+str(i))
            q = plt.quiver(xave, yave, dev_x[:,i], dev_y[:,i], scale=0.5)

            plt.quiverkey(q, 6500, 1500, 0.0238, label='1 mas', color='red', coordinates='data')
            plt.xlim(0,7000)
            plt.ylim(0,6500)
            plt.axes().set_aspect('equal')
            plt.ylabel("Y (pixels)")
            plt.xlabel("x (pixels)")
            plt.legend(loc='upper left')

            
   
    plt.figure(i+2)
    plt.title('Mean Translation Offsets')
    plt.plot(dx, 'o', label='x')
    plt.plot(dy, 'o', label='y')
    plt.ylabel('Offset (pix)')
    plt.legend(loc='upper left')

    plt.show()


    


def diff_psf(cbr=0.01):
    psff = glob.glob('*_psf.fits')
    #make differences fom first psf to the rest and save them

    fwhm_x = []
    fwhm_y = []
    beta = []
    amp = []
    ref = fits.getdata(psff[0])
    _res, _fwhm_x, _fwhm_y, _amp, _beta = fit_moffat(ref)
    fwhm_x.append(_fwhm_x)
    fwhm_y.append(_fwhm_y)
    beta.append(_beta)
    amp.append(_amp)
    
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    plt.figure(3)
    plt.clf()
    
    plt.subplot(4,len(psff), 1)
    plt.imshow(ref, vmin=0.0, vmax=0.06)
    plt.colorbar()
    plt.title('Frame 1 (ref)')

    plt.subplot(4, len(psff),2* len(psff) +1)
    plt.imshow((_res-ref)*-1.0, vmin=0, vmax=0.06)
    plt.gray()
    plt.colorbar()
    plt.title(" Moffat Fit")

    plt.subplot(4, len(psff),3* len(psff) +1)
    plt.imshow(_res, vmin=-0.006, vmax=0.006)
    plt.gray()
    plt.colorbar()
    plt.title("PSF - Moffat Fit")

    
    
    for i in range(len(psff)-1):
        _tmp = fits.getdata(psff[i+1])
        _diff = _tmp - ref
        plt.subplot(4, len(psff), i+2)
        plt.imshow(_tmp, vmin=0.0, vmax=0.06)
        plt.title('Frame '+str(i+2))
        plt.colorbar()

        _res, _fwhm_x, _fwhm_y, _amp, _beta = fit_moffat(_tmp)
        fwhm_x.append(_fwhm_x)
        fwhm_y.append(_fwhm_y)
        beta.append(_beta)
        amp.append(_amp)
    
        plt.subplot(4, len(psff), 2*len(psff) + i+2)
        plt.imshow(-1.0*(_res - _tmp ), vmin=-0.00, vmax=0.06)
        plt.gray()
        plt.colorbar()
        plt.title("Moffat Model")

        plt.subplot(4, len(psff), 1*len(psff) +  i+2)
        plt.imshow(_diff, vmin=-.006, vmax=0.006)
        plt.gray()
        plt.colorbar()
        plt.title(r"$ \Delta$")

        #plt.subplot(4, len(psff), 2*len(psff) +  i+2)
        #plt.imshow(_diff/(np.abs(ref) + np.abs(_tmp))**0.5, vmin=-.05, vmax=.05)
        #plt.gray()
        #plt.colorbar()
        #plt.title(r"$\Delta$ / $\sigma_{photon}$")

        plt.subplot(4, len(psff),3* len(psff) +  i+2)
        plt.imshow(_res, vmin=-.006, vmax=0.006)
        plt.gray()
        plt.colorbar()
        plt.title("PSF - Moffat Fit")
        
        
    plt.tight_layout()
    plt.figure(4)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(fwhm_x, 'o', label='x')
    plt.plot(fwhm_y, 'o', label='y')
    plt.title('FWHM')
    plt.ylabel('FWHM (pix)')
    
    plt.subplot(3, 1, 2)
    plt.plot(amp, 'o')
    plt.title('Amplitude')

    plt.subplot(3,1,3)
    plt.plot(beta, 'o')
    plt.title('Beta')
    plt.tight_layout()
    
    
def fit_moffat(data):
    '''
    '''

    y, x = np.mgrid[:data.shape[0], :data.shape[1]]                                                                                    
    z = data                                                                                                                 
    m_init = Elliptical_Moffat2D(N_sky = 0, amplitude=np.amax(z),  x_0=data.shape[0]/2.0, y_0=data.shape[1]/2.0, width_x = 2.0, width_y=2.0 )                                                                                                                                
    fit_m = fitting.LevMarLSQFitter()                                                                                              
    m = fit_m(m_init, x, y, z)                                                                                                     
                                                                                                                                               
    beta = m.power.value                                                                                                           
    fwhm_x = 2 * m.width_x.value * np.sqrt((2**(1/beta))-1)                                                                        
    fwhm_y = 2 * m.width_y.value * np.sqrt((2**(1/beta))-1)                                                                        
    amp = m.amplitude.value                                                                                                        
    res = z - m(x, y)
    return res, fwhm_x, fwhm_y, amp, beta

@custom_model
def Elliptical_Moffat2D(x, y, N_sky = 0., amplitude = 1., phi=0., power = 1., x_0 = 0., y_0 = 0., width_x = 1., width_y = 1.):                                                                       
    """                                                                                                                                       \
                                                                                                                                               
    A custom astropy model for a two dimensional elliptical moffat function.                                                                  \
                                                                                                                                               
    N_sky: a constant background value                                                                                                        \
                                                                                                                                               
    Amplitude: A                                                                                                                              \
                                                                                                                                               
    phi: rotation angle (in radians?)                                                                                                         \
                                                                                                                                               
    power: slope of model (beta)                                                                                                              \
                                                                                                                                               
    x_0, y_0: star's centroid coordinates                                                                                                     \
                                                                                                                                               
    width_x, width_y: core widths (alpha)                                                                                                     \
                                                                                                                                               
    """                                                                                                                                        
                                                                                                                                               
    c = np.cos(phi)                                                                                                                            
    s = np.sin(phi)                                                                                                                            
    A = (c / width_x) ** 2 + (s / width_y)**2                                                                                                  
    B = (s / width_x) ** 2 + (c/ width_y)**2                                                                                                   
    C = 2 * s * c * (1/width_x**2 - 1/width_y**2)                                                                                              
    denom = (1 + A * (x-x_0)**2 + B * (y-y_0)**2 + C*(x-x_0)*(y-y_0))**power                                                                   
                                                                                                                                               
    return N_sky + amplitude / denom             
