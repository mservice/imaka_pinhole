from imaka_pinhole import analyze_stack, match_pin
import numpy as np
import scipy

def test_var():
    #create reference "positoins"
    nframes = 400
    nstars = 30
    _xs = np.linspace(0, 5000, num=nstars)
    _ys = np.linspace(0, 5000, num=nstars)
    coo = np.meshgrid(_xs, _ys)
    _x = coo[0].flatten()
    _y = coo[1].flatten()
    x = np.zeros((len(_x), nframes))
    y = np.zeros((len(_y), nframes))
    for i in range(nframes):
        x[:,i] = _x
        y[:,i] = _y
        
    _norm = scipy.stats.norm(loc=0 , scale=0.3)
    #_norm = scipy.stats.norm(loc=0 , scale=1.0/6.0)
    err = _norm.rvs(len(x.flatten()))
    x = x + np.reshape(err, x.shape)
    err = _norm.rvs(len(y.flatten()))
    y = y + np.reshape(err, y.shape)
    #import pdb;pdb.set_trace()
    #nim, avartot, avartot_err, stdtot, stdtot_err = match_pin.calc_var_n(x)

    xncut2, yncut2, dum, dummer = analyze_stack.plot_ind_shift(x, y, x)
    analyze_stack.pix_phase(xncut2, yncut2, dum, dummer)
    #plt.figure(18)
    #plt.clf()
    #plt.loglog(nim, avartot, label='allan 100')
    #plt.loglog(nim, stdtot, label='rms 100')


    #x  = np.ones((600,20))
    #_norm = scipy.stats.norm(loc=0 , scale=1.0/6.0)
    #err = _norm.rvs(len(x.flatten()))
    #x = x + np.reshape(err, x.shape)
    #import pdb;pdb.set_trace()
    #nim, avartot, avartot_err, stdtot, stdtot_err = calc_var_n(x)

    
   
    #plt.loglog(nim, avartot, label='allan 600')
    #plt.loglog(nim, stdtot, label='rms 600')
    
    #plt.loglog(nim, 1.0/np.array(nim)**0.5/6.0, color='black')
    #plt.legend(loc='upper right')
    #plt.title('Deviation Test')
    #plt.xlabel('N')
    #plt.ylabel('Deviation (pixels)')
    #plt.tight_layout()
    #plt.savefig('var.png')


def sim_fit():

    _space - np.linspace(0, 42 * 168, num=42)
    xpin, ypin = np.meshgrid(_space, _space)

    #now put random errors on each spot...

    _norm = scipy.stats.norm(loc=0 , scale=0.3)
    #_norm = scipy.stats.norm(loc=0 , scale=1.0/6.0)
    err = _norm.rvs(len(xpin.flatten()))
    xpin = xpin + np.reshape(err, xpin.shape)
    err = _norm.rvs(len(ypin.flatten()))
    ypin = ypin + np.reshape(err, ypin.shape)

    #now we need to apply seperate high order stanformations to each stack.
    #we should also only sample a subregion of the FOV...
