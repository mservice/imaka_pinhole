import numpy as np
from astropy.table import Table
from . import scipy_FitHypersphere as fitsphere
import numpy.random as rnd
from . import rigid_transform, fit_cyl
from scipy import optimize #import leastsq
from jlu.util import statsIter


def mk_grid_sim():
    '''
    runs cmmm sim with different pos_err and offset size
    then keeps vlaues and makes a table
    '''

    offsets = np.linspace(1, 2, num=11)
    errors = np.linspace(0, .1, num=5)
    mean_off = []
    mean_off_rat = []
    off_col = []
    err_col = []
    mean_scatter = []
    std = []
    for off in offsets:
        for err in errors:
            _dx, _dy, _dz, _dxp, _dyp, _dzp, xstd, ystd, zstd = sim_all(pos_err = err, offsets=off, trials=1000)
            _std = np.std(np.concatenate((_dx, _dy, _dz)))
            _abs_mean = np.mean(np.abs(np.concatenate((_dx, _dy, _dz))))
            _mean_rat =  np.mean(np.abs(np.concatenate((_dxp, _dyp, _dzp))))
            _mean_err = np.mean(np.concatenate((xstd, ystd, zstd)))
            mean_off.append(_abs_mean)
            mean_off_rat.append(_mean_rat)
            mean_scatter.append(_mean_err)
            std.append(_std)
            
            off_col.append(off)
            err_col.append(err)


    #_out = Table(data=[off_col, err_col,mean_off ,std, mean_off_rat, mean_scatter], names=['offset (mm)', 'measurement error (mm)', 'mean abolsute residual (mm)', 'RMS of residuals (mm)', 'RMS of res/actual', 'mean cmm scatter (mm)'])
    _out = Table(data=[off_col, err_col,mean_off , mean_off_rat, mean_scatter], names=['offset (mm)', 'measurement error (mm)', 'mean abolsute residual (mm)', 'RMS of res/actual', 'mean cmm scatter (mm)'])
    _out.write('mc_err.txt', format='ascii.fixed_width')
            
def sim_all(ref_file='/Users/service/Imaka/Imaka_cmm/05112016/StructureCoordinates_zero.txt', trials=10000, pos_err=.1, offsets=0.765):
    '''
    MC sim to test how accurate measured strucutre coordinates will be.
    inputs:
    pos_err: INT, random error in each CMM measurement in mm
    offets: offsets with repect to CAD model coordiantes are drawn from a uniform distribution [-offsets:offsets] for x,y,z for each points
    trials: number of instantiaiotns to generate statistics
    '''

    #first assign which points are measrued in the same set
    #NOTE --> need to include telescope bushings in the second group
    reftab = Table.read(ref_file, format='ascii')
    points = [ ['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N','DME', 'DMS', 'DMW'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N'],['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N','DME', 'DMS', 'DMW'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N']]
    
    pall = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            pall.append(points[i][j])
    pname = np.unique(pall)
    _qin = []
    #import pdb;pdb.set_trace()
    for i in range(len(reftab)):
        if reftab['col1'][i] in pname:
            _qin.append(i)
    reftab = reftab[_qin]
    #convert points names to indics in the reference coordianates!!!
    dx = []
    dxp = []
    dy = []
    dyp = []
    dz = []
    dzp = []
    xstd = []
    ystd = []
    zstd = []
    p_ind = []
    #import pdb;pdb.set_trace()
    for i in range(len(points)):
        _ind = []
        for j in range(len(points[i])):
            #do actions here
            match = False
            k = 0
            
            while not match:
                if reftab['col1'][k] == points[i][j]:
                    match = True
                    _ind.append(k)
                else:
                    k +=1
        p_ind.append(_ind)
                    
            

   

    #first draw offsets, then apply to create as built coordiantes
    for tri in range(trials):

        xoff = []
        yoff = []
        zoff = []
        for xx in range(len(reftab)/3):
            _tempx = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            _tempy = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            _tempz = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            for yy in range(3):
                xoff.append(_tempx)
                yoff.append(_tempy)
                zoff.append(_tempz)
        xoff = np.array(xoff)
        yoff = np.array(yoff)
        zoff = np.array(zoff)
        #import pdb;pdb.set_trace()

       
        refmat = np.matrix([reftab['col2'], reftab['col3'], reftab['col4']]).T
        xact = reftab['col2'] +xoff
        yact = reftab['col3'] +yoff
        zact = reftab['col4'] +zoff

        xstack = np.zeros((len(points),len(reftab)))
        ystack = np.zeros((len(points),len(reftab)))
        zstack = np.zeros((len(points),len(reftab)))
       
        
        #now create the measured catalogs
        for pp in range(len(points)):
            xerr = rnd.normal(size=len(reftab)) * pos_err
            yerr = rnd.normal(size=len(reftab)) * pos_err
            zerr = rnd.normal(size=len(reftab)) * pos_err

            xm = xact + xerr
            ym = yact + yerr
            zm = zact + zerr

            #now need to apply a lare offset to put measured coordiantes into "cmm" frame
            _xoff = rnd.uniform(-500, 500, size=1)
            _yoff = rnd.uniform(-500, 500, size=1)
            _zoff = rnd.uniform(-500, 500, size=1)

            cmm_points = np.matrix([xm,ym,zm])
            
            cmm_points[0] += _xoff
            cmm_points[1] += _yoff
            cmm_points[2] += _zoff

            #now apply a radom rotation about a random vector
            uv = rnd.uniform(-1,1, size=3)
            uv = uv / (np.sqrt(np.sum(uv**2)))
            #check that vector has length 1
            assert uv[0]**2 + uv[1]**2 + uv[2]**2 < 1.05 and uv[0]**2 + uv[1]**2 + uv[2]**2 > 0.95

            ang = rnd.uniform(0,2*np.pi, size=1)
            cosa = np.cos(ang)
            sina = np.sin(ang)
            #import pdb;pdb.set_trace()
            R = np.matrix(np.array([[cosa + uv[0]**2*(1-cosa), uv[0]*uv[1]*(1-cosa) + uv[2] * sina, uv[0]*uv[2]*(1-cosa) - uv[1]*sina]
                                    , [uv[0]*uv[1]*(1-cosa) - uv[2] *sina, cosa + uv[1]**2*(1-cosa), uv[1]*uv[2]*(1-cosa)+uv[0]*sina]
                                    , [uv[0]*uv[2] *(1-cosa) + uv[1]*sina, uv[1]*uv[2]*(1-cosa) - uv[0]*sina, cosa + uv[2]**2*(1-cosa)]]))
            cmm_final = R * cmm_points
            
            #now cut down the measured coordinates to only the correct ones
            cmm_measured = cmm_final[:,p_ind[pp]]

            #now I have the cmm points, need to fit rigid tranfomraiton back to the original reference data, and compare "measured" to the expected points (xact,yact,zact)

            R, t = rigid_transform.rigid_transform_3D(cmm_measured.T, refmat[p_ind[pp],:])
            newcoo = R * cmm_measured + t

           
            xstack[pp,p_ind[pp]] = newcoo[0]
            ystack[pp,p_ind[pp]] = newcoo[1]
            zstack[pp,p_ind[pp]] = newcoo[2]

            #import pdb;pdb.set_trace()

        #make averages for these measurments
        xave = []
        yave = []
        zave = []
        _xstd = []
        _ystd = []
        _zstd = []
        
        multiple = True
        for kk in range(xstack.shape[1]):
            xave.append(np.mean(xstack[:,kk][xstack[:,kk] != 0]))
            yave.append(np.mean(ystack[:,kk][ystack[:,kk] != 0]))
            zave.append(np.mean(zstack[:,kk][zstack[:,kk] != 0]))
            std = True
            while multiple and std:
                if np.sum(xstack[:,kk] != 0) > 1:
                    _xstd.append(np.std(xstack[:,kk][xstack[:,kk] != 0]))
                    _ystd.append(np.std(ystack[:,kk][ystack[:,kk] != 0]))
                    _zstd.append(np.std(zstack[:,kk][zstack[:,kk] != 0]))
                    std = False
                else:
                    multiple=False
                

        xave = np.array(xave)
        yave = np.array(yave)
        zave = np.array(zave)
        if multiple:
            _xstd = np.array(_xstd)
            _ystd = np.array(_ystd)
            _zstd = np.array(_zstd)

            xstd.append(_xstd)
            ystd.append(_ystd)
            zstd.append(_zstd)

        #import pdb;pdb.set_trace()
        #want to know the measured offsets per point wrt the input offsets
        _dx = xave - xact
        _dxp = (np.abs(_dx) - np.abs(xoff)) / xoff
        _dy = yave - yact
        _dyp = (np.abs(_dy) - np.abs(yoff)) / yoff
        _dz = zave - zact
        _dzp = (np.abs(_dz) - np.abs(zoff)) / zoff

        #import pdb;pdb.set_trace()
        dx.append(_dx)
        dxp.append(_dxp)
        dy.append(_dy)
        dyp.append(_dyp)
        dz.append(_dz)
        dzp.append(_dzp)
        #import pdb;pdb.set_trace()

    if multiple:
        return np.array(dx), np.array(dy), np.array(dz), np.array(dxp), np.array(dyp), np.array(dzp), xstd, ystd, zstd
    else:
        return np.array(dx), np.array(dy), np.array(dz), np.array(dxp), np.array(dyp), np.array(dzp)

        

def sim_all_new_model(ref_file='/Users/service/Imaka/Imaka_cmm/05112016/StructureCoordinates_zero.txt', trials=10000, pos_err=.1, offsets=0.765):
    '''
    MC sim to test how accurate measured strucutre coordinates will be.
    inputs:
    pos_err: INT, random error in each CMM measurement in mm
    offets: offsets with repect to CAD model coordiantes are drawn from a uniform distribution [-offsets:offsets] for x,y,z for each points
    trials: number of instantiaiotns to generate statistics
    '''

    #first assign which points are measrued in the same set
    #NOTE --> need to include telescope bushings in the second group
    reftab = Table.read(ref_file, format='ascii')
    points = [ ['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N','DME', 'DMS', 'DMW'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N'],['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N','DME', 'DMS', 'DMW'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N']]
    
    pall = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            pall.append(points[i][j])
    pname = np.unique(pall)
    _qin = []
    #import pdb;pdb.set_trace()
    for i in range(len(reftab)):
        if reftab['col1'][i] in pname:
            _qin.append(i)
    reftab = reftab[_qin]
    #convert points names to indics in the reference coordianates!!!
    dx = []
    dxp = []
    dy = []
    dyp = []
    dz = []
    dzp = []
    xstd = []
    ystd = []
    zstd = []
    p_ind = []
    #import pdb;pdb.set_trace()
    for i in range(len(points)):
        _ind = []
        for j in range(len(points[i])):
            #do actions here
            match = False
            k = 0
            
            while not match:
                if reftab['col1'][k] == points[i][j]:
                    match = True
                    _ind.append(k)
                else:
                    k +=1
        p_ind.append(_ind)
                    
            

   

    #first draw offsets, then apply to create as built coordiantes
    for tri in range(trials):

        xoff = []
        yoff = []
        zoff = []
        off_dict = {}
        for xx in range(len(reftab)/3):
            _tempx = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            _tempy = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            _tempz = rnd.uniform(low=-1.0*offsets, high=offsets, size=1)[0]
            
            for yy in range(3):
                xoff.append(_tempx)
                yoff.append(_tempy)
                zoff.append(_tempz)
                off_dict[xx*3 + yy] = xx 
        xoff = np.array(xoff)
        yoff = np.array(yoff)
        zoff = np.array(zoff)
        #import pdb;pdb.set_trace()

       
        refmat = np.matrix([reftab['col2'], reftab['col3'], reftab['col4']]).T
        xact = reftab['col2'] +xoff
        yact = reftab['col3'] +yoff
        zact = reftab['col4'] +zoff

        xstack = np.zeros((len(points),len(reftab)))
        ystack = np.zeros((len(points),len(reftab)))
        zstack = np.zeros((len(points),len(reftab)))
       
        
        #now create the measured catalogs
        for pp in range(len(points)):
            xerr = rnd.normal(size=len(reftab)) * pos_err
            yerr = rnd.normal(size=len(reftab)) * pos_err
            zerr = rnd.normal(size=len(reftab)) * pos_err

            xm = xact + xerr
            ym = yact + yerr
            zm = zact + zerr

            #now need to apply a lare offset to put measured coordiantes into "cmm" frame
            _xoff = rnd.uniform(-500, 500, size=1)
            _yoff = rnd.uniform(-500, 500, size=1)
            _zoff = rnd.uniform(-500, 500, size=1)

            cmm_points = np.matrix([xm,ym,zm])
            
            cmm_points[0] += _xoff
            cmm_points[1] += _yoff
            cmm_points[2] += _zoff

            #now apply a radom rotation about a random vector
            uv = rnd.uniform(-1,1, size=3)
            uv = uv / (np.sqrt(np.sum(uv**2)))
            #check that vector has length 1
            assert uv[0]**2 + uv[1]**2 + uv[2]**2 < 1.05 and uv[0]**2 + uv[1]**2 + uv[2]**2 > 0.95

            ang = rnd.uniform(0,2*np.pi, size=1)
            cosa = np.cos(ang)
            sina = np.sin(ang)
            #import pdb;pdb.set_trace()
            R = np.matrix(np.array([[cosa + uv[0]**2*(1-cosa), uv[0]*uv[1]*(1-cosa) + uv[2] * sina, uv[0]*uv[2]*(1-cosa) - uv[1]*sina]
                                    , [uv[0]*uv[1]*(1-cosa) - uv[2] *sina, cosa + uv[1]**2*(1-cosa), uv[1]*uv[2]*(1-cosa)+uv[0]*sina]
                                    , [uv[0]*uv[2] *(1-cosa) + uv[1]*sina, uv[1]*uv[2]*(1-cosa) - uv[0]*sina, cosa + uv[2]**2*(1-cosa)]]))
            cmm_final = R * cmm_points
            
            #now cut down the measured coordinates to only the correct ones
            cmm_measured = cmm_final[:,p_ind[pp]]

            #now I have the cmm points, need to fit rigid tranfomraiton back to the original reference data, and compare "measured" to the expected points (xact,yact,zact)

            measured_off = fit_rigid_off(cmm_measured[0], cmm_measured[1], cmm_measured[2],reftab['col2'][p_ind[pp]], reftab['col3'][p_ind[pp]], reftab['col4'][p_ind[pp]], off_dict)
            
           
            _tmpoff = [[],[],[]]
            for gg in range(measured_off.shape[1]):
                for ff in range(measured_off.shape[0]):
                    for hh in range(3):
                        _tmpoff[ff].append(measured_off[ff][gg])

            measured_off = np.array(_tmpoff)
            xstack[pp,p_ind[pp]] = measured_off[0]
            ystack[pp,p_ind[pp]] = measured_off[1]
            zstack[pp,p_ind[pp]] = measured_off[2]

            #import pdb;pdb.set_trace()

        #make averages for these measurments
        #import pdb;pdb.set_trace()
        xave = []
        yave = []
        zave = []
        _xstd = []
        _ystd = []
        _zstd = []
        
        multiple = True
        for kk in range(xstack.shape[1]):
            xave.append(np.mean(xstack[:,kk][xstack[:,kk] != 0]))
            yave.append(np.mean(ystack[:,kk][ystack[:,kk] != 0]))
            zave.append(np.mean(zstack[:,kk][zstack[:,kk] != 0]))
            std = True
            while multiple and std:
                if np.sum(xstack[:,kk] != 0) > 1:
                    _xstd.append(np.std(xstack[:,kk][xstack[:,kk] != 0]))
                    _ystd.append(np.std(ystack[:,kk][ystack[:,kk] != 0]))
                    _zstd.append(np.std(zstack[:,kk][zstack[:,kk] != 0]))
                    std = False
                else:
                    multiple=False
                

        xave = np.array(xave)
        yave = np.array(yave)
        zave = np.array(zave)
        import pdb;pdb.set_trace()
        if multiple:
            _xstd = np.array(_xstd)
            _ystd = np.array(_ystd)
            _zstd = np.array(_zstd)

            xstd.append(_xstd)
            ystd.append(_ystd)
            zstd.append(_zstd)

        #import pdb;pdb.set_trace()
        #want to know the measured offsets per point wrt the input offsets
        _dx = xave - xact
        _dxp = (np.abs(_dx) - np.abs(xoff)) / xoff
        _dy = yave - yact
        _dyp = (np.abs(_dy) - np.abs(yoff)) / yoff
        _dz = zave - zact
        _dzp = (np.abs(_dz) - np.abs(zoff)) / zoff

        #import pdb;pdb.set_trace()
        dx.append(_dx)
        dxp.append(_dxp)
        dy.append(_dy)
        dyp.append(_dyp)
        dz.append(_dz)
        dzp.append(_dzp)
        #import pdb;pdb.set_trace()

    if multiple:
        return np.array(dx), np.array(dy), np.array(dz), np.array(dxp), np.array(dyp), np.array(dzp), xstd, ystd, zstd
    else:
        return np.array(dx), np.array(dy), np.array(dz), np.array(dxp), np.array(dyp), np.array(dzp)

def sim_all_2(ref_file='/Users/service/Imaka/Imaka_cmm/05112016/StructureCoordinates_zero.txt', trials=10000, pos_err=.1, offsets=0.765, cmm_off=1000):
    '''
    MC sim to test how accurate measured strucutre coordinates will be.  Same as 1, except that inst3ad of aligning the individual data sets to the reference, we first collapse the data sets to a common frame
    inputs:
    pos_err: INT, random error in each CMM measurement in mm
    offets: offsets with repect to CAD model coordiantes are drawn from a uniform distribution [-offsets:offsets] for x,y,z for each points
    trials: number of instantiaiotns to generate statistics
    '''

    #first assign which points are measrued in the same set
    #NOTE --> need to include telescope bushings in the second group
    reftab = Table.read(ref_file, format='ascii')
    points = [['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N', 'DME'], ['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N', 'DME'], ['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N'],['AOM1E', 'AOM1N', 'AOM1S', 'AOM4N', 'AOM4E', 'AOM4S', 'AOM2S', 'AOM2W', 'AOM2N', 'DME'], ['AOM3E', 'AOM3N', 'AOM3S','AOM2S', 'AOM2W', 'AOM2N']]

    num_fail = 0
    
    pall = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            pall.append(points[i][j])
    pname = np.unique(pall)
    _qin = []
    #import pdb;pdb.set_trace()
    for i in range(len(reftab)):
        if reftab['col1'][i] in pname:
            _qin.append(i)
    reftab = reftab[_qin]
    #convert points names to indics in the reference coordianates!!!
    dx = []
    dy = []
    dz = []
    xstd = []
    ystd = []
    zstd = []
    p_ind = []
    
    #import pdb;pdb.set_trace()
    for i in range(len(points)):
        _ind = []
        for j in range(len(points[i])):
            #do actions here
            match = False
            k = 0
            
            while not match:
                if reftab['col1'][k] == points[i][j]:
                    match = True
                    _ind.append(k)
                else:
                    k +=1
        p_ind.append(_ind)
                    
            

   

    #first draw offsets, then apply to create as built coordiantes
    for i in range(trials):
        xoff = rnd.uniform(low=-1.0*offsets, high=offsets, size=len(reftab))
        yoff = rnd.uniform(low=-1.0*offsets, high=offsets, size=len(reftab))
        zoff = rnd.uniform(low=-1.0*offsets, high=offsets, size=len(reftab))

       
        refmat = np.matrix([reftab['col2'], reftab['col3'], reftab['col4']]).T
        xact = reftab['col2'] +xoff
        yact = reftab['col3'] +yoff
        zact = reftab['col4'] +zoff

        xstack = np.zeros((len(points),len(reftab)))
        ystack = np.zeros((len(points),len(reftab)))
        zstack = np.zeros((len(points),len(reftab)))

        cmm_all = []
        cmm_debug = []
        
        #now create the measured catalogs
        for pp in range(len(points)):
            xerr = rnd.normal(size=len(reftab)) * pos_err
            yerr = rnd.normal(size=len(reftab)) * pos_err
            zerr = rnd.normal(size=len(reftab)) * pos_err

            xm = xact + xerr
            ym = yact + yerr
            zm = zact + zerr

            #now need to apply a large offset to put measured coordiantes into "cmm" frame
            _xoff = rnd.uniform(-1.0*cmm_off, cmm_off, size=1)
            _yoff = rnd.uniform(-1.0*cmm_off, cmm_off, size=1)
            _zoff = rnd.uniform(-1.0*cmm_off, cmm_off, size=1)

            cmm_points = np.matrix([xm,ym,zm])
            
            cmm_points[0] += _xoff
            cmm_points[1] += _yoff
            cmm_points[2] += _zoff

            #now apply a radom rotation about a random vector
            uv = rnd.uniform(-1,1, size=3)
            uv = uv / (np.sqrt(np.sum(uv**2)))
            #check that vector has length 1
            assert uv[0]**2 + uv[1]**2 + uv[2]**2 < 1.05 and uv[0]**2 + uv[1]**2 + uv[2]**2 > 0.95

            ang = rnd.uniform(0,2*np.pi, size=1)
            cosa = np.cos(ang)
            sina = np.sin(ang)
            #import pdb;pdb.set_trace()
            R = np.matrix(np.array([[cosa + uv[0]**2*(1-cosa), uv[0]*uv[1]*(1-cosa) + uv[2] * sina, uv[0]*uv[2]*(1-cosa) - uv[1]*sina]
                                    , [uv[0]*uv[1]*(1-cosa) - uv[2] *sina, cosa + uv[1]**2*(1-cosa), uv[1]*uv[2]*(1-cosa)+uv[0]*sina]
                                    , [uv[0]*uv[2] *(1-cosa) + uv[1]*sina, uv[1]*uv[2]*(1-cosa) - uv[0]*sina, cosa + uv[2]**2*(1-cosa)]]))
            cmm_final = R * cmm_points
            assert np.abs(1 - np.linalg.det(R)) < .001
            #import pdb;pdb.set_trace()
            
            #now cut down the measured coordinates to only the correct ones
            #cmm_measured = cmm_final[:,p_ind[pp]]
            cmm_all.append(cmm_final)
            cmm_debug.append(np.matrix([xm,ym,zm])[:,p_ind[pp]])

        #now need to find common points between the data sets
        cmm_ind = []
        #import pdb;pdb.set_trace()
        for ii in range(len(p_ind)-1):
            cmm_ind.append({})
            for jj in range(len(p_ind[0])):
                for kk in range(len(p_ind[ii+1])):
                    if p_ind[0][jj] == p_ind[ii+1][kk]:
                        cmm_ind[-1][jj] = kk
                        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        #now we have a list of dictionaries (with length of total data sets - 1) matching indixes from the first point list to each of the other point lists
        cmm_align = []
        xstack[0,p_ind[0]] = cmm_all[0][0]
        ystack[0,p_ind[0]] = cmm_all[0][1]
        zstack[0,p_ind[0]] = cmm_all[0][2]
        for ii in range(len(cmm_all)-1):
            refind = cmm_ind[0].keys()
            _tmp_in = []
            for jj in range(len(refind)):
                
                _tmp_in.append(cmm_ind[0][refind[jj]])
                #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            R, t = rigid_transform.rigid_transform_3D(cmm_all[ii+1].T[_tmp_in,:], cmm_all[0].T[refind,:])
            _new = R * cmm_all[ii+1] + t
            if np.mean(np.abs((_new[:,_tmp_in] - cmm_all[0][:,refind]).flatten())) < 5:
                num_fail +=1
            rdistN = np.sqrt((_new[:,_tmp_in][0,0] - _new[:,_tmp_in][1,0])**2 + (_new[:,_tmp_in][0,1] - _new[:,_tmp_in][1,1])**2 + (_new[:,_tmp_in][0,2] - _new[:,_tmp_in][1,2])**2)
            rdistNC = np.sqrt((cmm_all[ii+1][:,_tmp_in][0,0] - cmm_all[ii+1][:,_tmp_in][1,0])**2 + (cmm_all[ii+1][:,_tmp_in][0,1] - cmm_all[ii+1][:,_tmp_in][1,1])**2 + (cmm_all[ii+1][:,_tmp_in][0,2] - cmm_all[ii+1][:,_tmp_in][1,2])**2)

            #rN = np.sqrt((_cmm_all[ii+1][:,_tmp_in][0,0] - _cmm_all[ii+1][:,_tmp_in][0])**2 + (cmm_All[ii+1][:,_tmp_in][1,0] - cmm_All[ii+1][:,_tmp_in][1])**2 + (cmm_all[ii+1][:,_tmp_in][2,0] - 
            rdistC = np.sqrt((cmm_all[0][:,refind][0,0] - cmm_all[0][:,refind][1,0])**2 + (cmm_all[0][:,refind][0,1] - cmm_all[0][:,refind][1,1])**2 + (cmm_all[0][:,refind][0,1] - cmm_all[0][:,refind][1,2])**2)
            assert np.abs(rdistN - rdistC) < 5
                
            #import pdb;pdb.set_trace()
            xstack[ii+1,p_ind[ii+1]] = _new[0]
            ystack[ii+1,p_ind[ii+1]] = _new[1]
            zstack[ii+1,p_ind[ii+1]] = _new[2]
            #import pdb;pdb.set_trace()
            
                

        #now I have the cmm points, need to fit rigis tranfomraiton back to the original reference data, and compare "measured" to the expected points (xact,yact,zact)


        #make averages for these measurments
        xave = []
        yave = []
        zave = []
        _xstd = []
        _ystd = []
        _zstd = []
        
        multiple = True
        for kk in range(xstack.shape[1]):
            xave.append(np.mean(xstack[:,kk][xstack[:,kk] != 0]))
            yave.append(np.mean(ystack[:,kk][ystack[:,kk] != 0]))
            zave.append(np.mean(zstack[:,kk][zstack[:,kk] != 0]))
            std = True
            while multiple and std:
                if np.sum(xstack[:,kk] != 0) > 1:
                    _xstd.append(np.std(xstack[:,kk][xstack[:,kk] != 0]))
                    _ystd.append(np.std(ystack[:,kk][ystack[:,kk] != 0]))
                    _zstd.append(np.std(zstack[:,kk][zstack[:,kk] != 0]))
                    std = False
                else:
                    multiple=False
                

        xave = np.array(xave)
        yave = np.array(yave)
        zave = np.array(zave)
        #now need to shift collated coordiantes to the final reference
        mat_ave = np.matrix([xave, yave, zave]).T
        R, t = rigid_transform.rigid_transform_3D(mat_ave, refmat)
        newcoo = R * mat_ave.T + t

        xave = newcoo[0]
        yave = newcoo[1]
        zave = newcoo[2]

       
        
        if multiple:
            _xstd = np.array(_xstd)
            _ystd = np.array(_ystd)
            _zstd = np.array(_zstd)

            xstd.append(_xstd)
            ystd.append(_ystd)
            zstd.append(_zstd)

        #want to know the measured offsets per point wrt the input offsets
        _dx = xave - xact
        _dy = yave - yact
        _dz = zave - zact

        dx.append(_dx)
        dy.append(_dy)
        dz.append(_dz)
        #import pdb;pdb.set_trace()

    print 'Number failed ', num_fail
    if multiple:
        return dx, dy, dz, xstd, ystd, zstd
    else:
        return np.array(dx), np.array(dy), np.array(dz)

            




def mk_rand_rot():
    '''
    returns random 3d rotation
    '''

    return R 

def sim_sphere(num_points=30, ang_size=0.2, r=1150, phi_min=0, noise=.05 ):

    #first randomly draw points
    vmin = (np.cos(phi_min) + 1) / 2.0
    vmax = (np.cos(ang_size) + 1) / 2.0
    u = np.random.uniform(size=num_points)
    v = np.random.uniform(low=vmax, high=vmin, size=num_points)
    theta = 2 * np.pi *u
    phi = np.arccos(2 * v - 1)
    

    #import pdb;pdb.set_trace()
    #now calculate cartesian coordiantes with addition of random error
    x = r * np.sin(theta) * np.cos(phi) + np.random.normal(size=num_points) * noise
    y = r * np.sin(theta) * np.sin(phi) + np.random.normal(size=num_points) * noise
    z = r * np.cos(theta)  +  np.random.normal(size=num_points) * noise

    data = np.array([x, y, z]).T
    sphere_param = fitsphere.fit_hypersphere(data)
    return sphere_param[0]



def mc_sphere(num_attempts=10000, r=1150, ang_size=0.2, noise=0.05, num_points=30):

    rad = []
    for i in range(num_attempts):
        _r = sim_sphere(r=r, ang_size=ang_size, noise=noise, num_points=num_points)
        rad.append(_r)

    rad = np.array(rad)
    rad = rad - r
    print 'scatter in radius measurments is ' + str(np.std(rad))
    
    
def sim_align():
    #want to know relative error in the final alignment assuming CMM coo used to do everything, with some # of reference on T int plate
    pass

def fit_sphere_AOM(infile = '/Users/service/Imaka/Imaka_cmm/mirrors/AOM3.txt'):

    dat = Table.read(infile, format='ascii')
    pbool = dat['col2'] == 'top_surface'

    indat = np.array([dat['col4'][pbool], dat['col5'][pbool], dat['col6'][pbool]]).T
    sphere_param = fitsphere.fit_hypersphere(indat)
    
    zeval = eval_sphere(sphere_param, dat['col4'][pbool], dat['col5'][pbool])
    zresid = dat['col6'][pbool] - zeval
    return sphere_param, zresid

def find_piston(tip_dia=3):
    sfile_3 = '/Users/service/Imaka/Imaka_cmm/mirrors/AOM3.txt'
    sfile_1 = '/Users/service/Imaka/Imaka_cmm/mirrors/AOM1.txt'
    all_f = '/Users/service/Imaka/Imaka_cmm/mirrors/mirrors_20160616.txt'

    #first find average point for posts
    mtab = Table.read(all_f, format='ascii')
    mbool = []
    pb = []
    pb3 = []
    ave_pos = []
    
    for i in range(3):
        pb.append(mtab['col3'] == 'post'+str(i+1)+'_ave')

    for i in range(3):
        pb3.append(mtab['col3'] == 'post'+str(i+1)+'A_ave')
    mbool.append(mtab['col1'] == 'AOM3')
    mbool.append(mtab['col1'] == 'AOM1')
    mbool.append(mtab['col1'] == 'AOM4')
    avex = (mtab['col4'][mbool[0]*pb3[0]]+mtab['col4'][mbool[0]*pb3[1]]+mtab['col4'][mbool[0]*pb3[2]])/3.0
    avey = (mtab['col5'][mbool[0]*pb3[0]]+mtab['col5'][mbool[0]*pb3[1]]+mtab['col5'][mbool[0]*pb3[2]])/3.0
    avez = (mtab['col6'][mbool[0]*pb3[0]]+mtab['col6'][mbool[0]*pb3[1]]+mtab['col6'][mbool[0]*pb3[2]])/3.0
    

    spar, zresid = fit_sphere_AOM(infile=sfile_3)

    #phi = np.linspace(0, np.pi, num=100)
    #theta = np.linspace(0,2*np.pi, num=100)
    phi, theta = np.mgrid[0.0:np.pi:1000j, 0.0:2.0*np.pi:1000j]
    x = spar[1][0] + spar[0]*np.cos(theta)*np.sin(phi)
    y = spar[1][1] + spar[0]*np.sin(theta)*np.sin(phi)
    z = spar[1][2] + spar[0]*np.cos(phi)
    #now find minimum distanve between spere and average point
    piston3 = np.min(np.sqrt((x-avex)**2 + (y - avey)**2 + (z - avez)**2)) - tip_dia / 2.0


    print 'using tip diameter of ', tip_dia
    print 'piston for AOM3 (mm)', piston3
    

    #import pdb;pdb.set_trace()
    spar, zresid = fit_sphere_AOM(infile=sfile_1)

    avex = (mtab['col4'][mbool[1]*pb[0]]+mtab['col4'][mbool[1]*pb[1]]+mtab['col4'][mbool[1]*pb[2]])/3.0
    avey = (mtab['col5'][mbool[1]*pb[0]]+mtab['col5'][mbool[1]*pb[1]]+mtab['col5'][mbool[1]*pb[2]])/3.0
    avez = (mtab['col6'][mbool[1]*pb[0]]+mtab['col6'][mbool[1]*pb[1]]+mtab['col6'][mbool[1]*pb[2]])/3.0

    x = spar[1][0] + spar[0]*np.cos(theta)*np.sin(phi)
    y = spar[1][1] + spar[0]*np.sin(theta)*np.sin(phi)
    z = spar[1][2] + spar[0]*np.cos(phi)

    piston1 = np.min(np.sqrt((x-avex)**2 + (y - avey)**2 + (z - avez)**2))  - tip_dia / 2.0
    print 'piston for AOM1 (mm)', piston1


    topbool = mtab['col2'] == 'top_surface'
    aom4 = Table.read('/Users/service/Imaka/Imaka_cmm/mirrors/AOM4.txt', format='ascii')
    points = np.zeros((3 , len(aom4)))
    for i in range(len(aom4)):
        points[0,i] = aom4['col1'][i]
        points[1,i] = aom4['col2'][i]
        points[2,i] = aom4['col3'][i]
    
    p, n = fit_cyl.planeFit(points)

    xe = np.linspace(p[0] - 500, p[0] + 500, num=100000)
    ye = np.linspace(p[1] - 500, p[1] + 500, num=100000)
    ze = fit_cyl.plane_eval(p, n, xe, ye)

    avex = (mtab['col4'][mbool[2]*pb3[0]]+mtab['col4'][mbool[2]*pb3[1]]+mtab['col4'][mbool[2]*pb3[2]])/3.0
    avey = (mtab['col5'][mbool[2]*pb3[0]]+mtab['col5'][mbool[2]*pb3[1]]+mtab['col5'][mbool[2]*pb3[2]])/3.0
    avez = (mtab['col6'][mbool[2]*pb3[0]]+mtab['col6'][mbool[2]*pb3[1]]+mtab['col6'][mbool[2]*pb3[2]])/3.0
    
    piston3 = np.min(np.sqrt((avex - xe)**2 + (avey - ye)**2 + (avez - ze)**2)) - tip_dia / 2.0
    print 'piston for AOM4 (mm)', piston3
    import pdb;pdb.set_trace()


def fit_bolt_circle(infile='imaka_telint.txt'):
    data= Table.read('imaka_telint.txt', format='ascii')

    cbool = data['col2']=='Circle Centers'
    pbool = (data['col2'] == 'telint_plane') + (data['col2'] == 'telint_planeA')

    plane = data[pbool]
    x = data['col4'][cbool]
    y = data['col5'][cbool]
    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)


    points = np.zeros((3,len(plane)))
    for i in range(len(plane)):
        points[0,i] = plane['col4'][i]
        points[1,i] = plane['col5'][i]
        points[2,i] = plane['col6'][i]
    p, n = fit_cyl.planeFit(points)

    #xe = np.linspace(p[0] - 500, p[0] + 500, num=100000)
    #ye = np.linspace(p[1] - 500, p[1] + 500, num=100000)
    ze = fit_cyl.plane_eval(p, n, xc_2, yc_2)
    print 'center of telescope interface is ', xc_2, yc_2, ze-1.5
    
    #import pdb;pdb.set_trace()
    #first fit circle to the circle centers, but only using the x and y coordiantes

def fit_bolt_circle2(infile='20160809_1031am.txt'):
    data= Table.read(infile, format='ascii')

    cbool = data['col2']=='bolt_circle'
    pbool = (data['col2'] == 'telint_plane') + (data['col2'] == 'telint_planeA') + (data['col2'] == 'telint1') + (data['col2'] == 'telint2')+ (data['col2'] == 'telint3') +(data['col2'] == 'telint')

    plane = data[pbool]
    x = data['col4'][cbool]
    y = data['col5'][cbool]
    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)


    points = np.zeros((3,len(plane)))
    for i in range(len(plane)):
        points[0,i] = plane['col4'][i]
        points[1,i] = plane['col5'][i]
        points[2,i] = plane['col6'][i]
    p, n = fit_cyl.planeFit(points)
    print 'normal vector for plane', n

    #xe = np.linspace(p[0] - 500, p[0] + 500, num=100000)
    #ye = np.linspace(p[1] - 500, p[1] + 500, num=100000)
    ze = fit_cyl.plane_eval(p, n, xc_2, yc_2)
    print 'center of telescope interface is ', xc_2, yc_2, ze-1.5

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    data= Table.read('20160809_1031am.txt', format='ascii')
    cbool = data['col2']=='bolt_circle'
    #pbool = (data['col2'] == 'telint_plane') + (data['col2'] == 'telint_planeA')

    x = data['col4'][cbool]
    y = data['col5'][cbool]
    #data= Table.read('imaka_telint.txt', format='ascii')

    #cbool = data['col2']=='Circle Centers'
    #pbool = (data['col2'] == 'telint_plane') + (data['col2'] == 'telint_planeA')

    #x = data['col4'][cbool]
    #y = data['col5'][cbool]
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def find_off_mirrors(infile='align_tel_telint.txt'):
    data= Table.read(infile, format='ascii')

    cbool = data['col2']=='bolt_circle'
    pbool = (data['col2'] == 'telint_plane') + (data['col2'] == 'telint_planeA') + (data['col2'] == 'telint1') + (data['col2'] == 'telint2')+ (data['col2'] == 'telint3')

    plane = data[pbool]
    x = data['col4'][cbool]
    y = data['col5'][cbool]
    x_m = np.mean(x)
    y_m = np.mean(y)

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)


    points = np.zeros((3,len(plane)))
    for i in range(len(plane)):
        points[0,i] = plane['col4'][i]
        points[1,i] = plane['col5'][i]
        points[2,i] = plane['col6'][i]
    p, n = fit_cyl.planeFit(points)
    print 'normal vector for plane', n

    #xe = np.linspace(p[0] - 500, p[0] + 500, num=100000)
    #ye = np.linspace(p[1] - 500, p[1] + 500, num=100000)
    ze = fit_cyl.plane_eval(p, n, xc_2, yc_2)
    tel_ent = (xc_2, yc_2, ze-1.5)
    print 'center of telescope interface is ', xc_2, yc_2, ze-1.5

    #now need to define plane that we want the optics to be on
    #we choose to use the fiducial ??
    fid_upb = np.zeros(len(data), dtype='bool')
    fid_Sb = np.zeros(len(data), dtype='bool')
    fid_Nb = np.zeros(len(data), dtype='bool')
    aom2b =  np.zeros(len(data), dtype='bool')
    dmb =  np.zeros(len(data), dtype='bool')
    for i in range(len(data)):
        if 'fid_up' in data['col2'][i]:
            fid_upb[i] = True
        if 'fid_S' in data['col2'][i]:
            fid_Sb[i] = True
        if 'fid_N' in data['col2'][i]:
            fid_Nb[i] = True
        if 'dm' in data['col2'][i]:
            dmb[i] = True
        if 'AOM2' in data['col2'][i]:
            aom2b[i] = True 
        

    fid_up = (np.mean(data['col4'][fid_upb]), np.mean(data['col5'][fid_upb]),np.mean(data['col6'][fid_upb]))
    fid_N = (np.mean(data['col4'][fid_Nb]), np.mean(data['col5'][fid_Nb]),np.mean(data['col6'][fid_Nb]))
    fid_S = (np.mean(data['col4'][fid_Sb]), np.mean(data['col5'][fid_Sb]),np.mean(data['col6'][fid_Sb]))
    aom2 = np.array((np.mean(data['col4'][aom2b]), np.mean(data['col5'][aom2b]),np.mean(data['col6'][aom2b])))
    dm = np.array((np.mean(data['col4'][dmb]), np.mean(data['col5'][dmb]),np.mean(data['col6'][dmb])))
    fi_points = np.array([fid_up, fid_N, fid_S]).T
    pf, nf = fit_cyl.planeFit(fi_points)
    #xt = np.linspace(-1000, 1000, num=1000000)
    #yt = np.linspace(-1000, 1000, num=1000000)
    #for aom2
    v = aom2 - tel_ent
    daom2 = np.dot(v, n)
    deltas = daom2 * n
    print 'aom2 must be moved', deltas
    v = dm - tel_ent
    ddm = np.dot(v, n)
    deltas = ddm * n
    print 'dm must be moved', deltas
    
    
    import pdb;pdb.set_trace()
    
    
    
        



def err_sphere_AOM(infile = '/Users/service/Imaka/Imaka_cmm/mirrors/AOM3.txt', mctrials=10000, noise=.03 ):
    dat = Table.read(infile, format='ascii')
    pbool = dat['col2'] == 'top_surface'
    num_points = np.sum(pbool)
    r = []

    for i in range(mctrials):
        ex = np.random.normal(size=num_points) * noise
        ey = np.random.normal(size=num_points) * noise
        ez = np.random.normal(size=num_points) * noise
        indat = np.array([dat['col4'][pbool]+ez, dat['col5'][pbool]+ey, dat['col6'][pbool]]+ez).T
        sphere_param = fitsphere.fit_hypersphere(indat)
        r.append(sphere_param[0])
    print 'scatter in r ', np.std(r)
    return r
        
    
def eval_sphere(sphere_param, x, y):
    #import pdb;pdb.set_trace()
    z = np.sqrt(sphere_param[0]**2 - (x - sphere_param[1][0])**2 - (y - sphere_param[1][1])**2) + sphere_param[1][2]
    return z



def fit_rigid_off(x1,y1,z1,x2,y2,z2, meth='min'):


    if meth == 'min':
        off_ar = np.zeros((3,x1.shape[0]/3))
        off_ar = off_ar.flatten()
        res = optimize.minimize(rigid_min, off_ar, args=(x1, y1, z1, x2, y2, z2), method='SLSQP')
        import pdb;pdb.set_trace()
        return np.reshape(res.x,(3,x1.shape[0]/3))
    elif meth=='grid':
        err = []
        off_ar = np.zeros((3,x1.shape[0]/3))
        off_ar = off_ar.flatten()
        off_guess = np.linspace(0, 1, num=5)
        #for i in range(len(off_ar.flatten())):
        #    for j in range(len(off_guess)):
        mesh = []
        for i in range(len(off_ar)):
            mesh.append(off_guess)
        guess = np.meshgrid(mesh)
        import pdb;pdb.set_trace()
                #off_ar[i] = off_guess[j]
                #_err = rigid_min(np.reshape(off_ar, (3,x1.shape[0]/3)), x1, y1, x1, x2, y2, z2)
                #xerr.append(_err)
    #import pdb;pdb.set_trace()
                
        #for i in range(len(off_guess)):
        #    off_ar[
        #    for j in range(len(off_guess)):
        #        for k in range(len(off_guess)):
        #            for zz in range(len(x1.shape[0]/3)):
                        
        
    
    
def rigid_min(off_ar, x1,y1,z1,x2,y2,z2):

    _dx = []
    _dy = []
    _dz = []
    off_ar = np.reshape(off_ar, (3,x1.shape[0]/3))
    for i in range(off_ar.shape[1]):
        for jj in range(3):
            _dx.append(off_ar[0][i])
            _dy.append(off_ar[1][i])
            _dz.append(off_ar[2][i])

    _dx = np.array(_dx)
    _dy = np.array(_dy)
    _dz = np.array(_dz)

    x2 = x2 + _dx
    y2 = y2 + _dy
    z2 = z2 + _dz
    #mat1 = np.matrix([np.array(x1)[0] - _dx, np.array(y1)[0]-_dy, np.array(z1)[0]]-_dz)
    mat1 = np.matrix([np.array(x1), np.array(y1), np.array(z1)])
    mat2 = np.matrix([x2,y2,z2])
    #import pdb;pdb.set_trace()
    R, t = rigid_transform.rigid_transform_3D(mat1.T, mat2.T)
    newcoo = np.array(R *mat1 + t)
    xn = newcoo[0]
    yn = newcoo[1]
    zn = newcoo[2]

    err = np.sum(np.sqrt((xn - x2)**2 + (yn - y2)**2 + (zn - z2)**2))
    #import pdb;pdb.set_trace()
    return err
        
    
    
def test_fit(fac=.1, meth='grid'):

    xr = np.array([  53.71693181,  367.32657847,  761.20285866,   20.95847269,
        560.87883803,  292.03716761,  837.3706314 ,  758.30412038,
        562.67117267,  301.01187134,   69.62770891,  595.33053462,
         49.54464293,  101.02613144,  368.01217439,  560.1804659 ,
        251.10731144,  611.14553755])
    
    yr = np.array([ 117.99785211,  685.83527725,  472.63524591,  984.46006714,
        684.55866728,  331.97981619,  790.27915497,   84.33582458,
        308.75758385,  460.90279395,  499.00007703,   23.15929271,
        953.09080386,  536.22291907,  226.95506455,   19.46479006,
        427.8007622 ,   69.37563665])

    zr = np.array([  32.68242787,  634.66186022,   81.19569334,  428.20626603,
        653.46845078,  496.13863153,  332.10980769,   30.2798893 ,
         12.71700097,  565.16481837,  923.31117401,  855.07732957,
        653.95636809,  831.57547211,  886.64052343,  367.91158242,
        887.84709901,318.4551511076263])

    xoff = np.array([ 0.17147857,   0.17147857,   0.17147857,  0.71776677,  0.71776677 ,
        0.71776677,  0.05285637, 0.05285637,  0.05285637,
        0.31320588,  0.31320588 ,  0.31320588,  0.93476223,
        0.93476223,  0.93476223,  0.13174679,  0.13174679,  0.13174679])*fac

    xot = np.array([.17114, .7177, .0528, .3132, .934, .131])*fac
    
    yoff = np.array([ 0.13039668, 0.13039668,  0.13039668,  0.59253225,  0.59253225,
        0.59253225,  0.1347229 ,  0.1347229,  0.1347229,  0.61804915,
        0.61804915, 0.61804915,  0.37571104,  0.37571104,  0.37571104,
        0.53285239,  0.53285239,  0.53285239])*fac
    yot = np.array([ 0.13039668,  0.59253225,  0.1347229,  0.61804915,  0.37571104,  0.53285239])*fac

    zoff = np.array([ 0.87481824,  0.87481824,  0.87481824,  0.92644118,  0.92644118,
        0.92644118,  0.28896515,  0.28896515,  0.28896515,  0.74229647,
        0.74229647,  0.74229647,  0.43209845,  0.43209845,  0.43209845,
        0.54205681, 0.54205681,  0.54205681])*fac
    zot = np.array([0.87481824,  0.92644118,  0.28896515,  0.74229647,  0.43209845, 0.54205681])*fac
    
    


    ang = np.pi/3
    cosa = np.cos(ang)
    sina = np.sin(ang)
    #import pdb;pdb.set_trace()
    uv = rnd.uniform(-1,1, size=3)
    uv = uv / (np.sqrt(np.sum(uv**2)))
    R = np.matrix(np.array([[cosa + uv[0]**2*(1-cosa), uv[0]*uv[1]*(1-cosa) + uv[2] * sina, uv[0]*uv[2]*(1-cosa) - uv[1]*sina]
                                    , [uv[0]*uv[1]*(1-cosa) - uv[2] *sina, cosa + uv[1]**2*(1-cosa), uv[1]*uv[2]*(1-cosa)+uv[0]*sina]
                                    , [uv[0]*uv[2] *(1-cosa) + uv[1]*sina, uv[1]*uv[2]*(1-cosa) - uv[0]*sina, cosa + uv[2]**2*(1-cosa)]]))
    new = np.matrix([xr+xoff,yr+yoff,zr+zoff])
    cmm_final = R * new
    xn = np.array(cmm_final[0])[0]+ 200
    yn = np.array(cmm_final[1])[0]+ 500
    zn = np.array(cmm_final[2])[0]- 1000

    #import pdb;pdb.set_trace()
    #first test that ifI put in the correct answer, get no errors
    off_cor = np.array([xot, yot, zot])
    e1 = rigid_min(off_cor, xn, yn, zn, xr, yr, zr)
    assert e1 < .01
    offsets = fit_rigid_off(xn,yn,zn,xr,yr,zr, meth=meth)
    print 'mean x residual', np.mean(np.abs(offsets[0]-xot))
    print 'mean y residual', np.mean(np.abs(offsets[1]-yot))
    print 'mean z residual', np.mean(np.abs(offsets[2]-zot))
    import pdb;pdb.set_trace()

    
    
def ave_fid(filename='20160809_1031am.txt', gname='fid_up1'):
    data = Table.read(filename, format='ascii')
    gbool = np.zeros(len(data), dtype='bool')
    for i in range(len(data)):
        if gname in data['col2'][i]:
            gbool[i] = True

    #now report values
    data = data[gbool]
    xerr = np.std(data['col4'])
    yerr = np.std(data['col5'])
    zerr = np.std(data['col6'])
    print 'X scatter ', xerr
    print 'Y scatter ', yerr
    print 'Z scatter ', zerr

    xave =  np.mean(data['col4'])
    yave =np.mean(data['col5'])
    zave = np.mean(data['col6'])
    print 'Mean position', xave, ' ',yave, ' ',zave
    sbool = (data['col4'] > xave - 3 * xerr) * (data['col4'] < xave + 3 * xerr) * (data['col5'] < yave + 3 * yerr) * (data['col5'] > yave - 3 * yerr) * (data['col6'] < zave + 3 * zerr) * (data['col6'] > zave - 3 * zerr)
    print np.sum(sbool), ' of ',  len(data), ' points used' 
    print 'Mean position w/3-sig clipping', np.mean(data['col4'][sbool]),' ', np.mean(data['col5'][sbool]), ' ', np.mean(data['col6'][sbool])
    #print 'Median position',  np.median(data['col4']), ' ', np.median(data['col5']), ' ', np.median(data['col6'])
    #import pdb;pdb.set_trace()

def ave_sig_clip(x, y, z):
    xave =  np.mean(x)
    yave =np.mean(y)
    zave = np.mean(z)

    xerr = np.std(x)
    yerr = np.std(y)
    zerr = np.std(z)
    sbool = (x > xave - 3 * xerr) * (x < xave +3 * xerr) *( y > yave - 3 * yerr) * (y < yave + 3 * yerr) * (z < zave + 3 * zerr) * (z > zave - 3 *zerr)
    print np.sum(sbool), ' of ',  len(x), ' points used'
    return np.array([np.mean(x[sbool]), np.mean(y[sbool]), np.mean(z[sbool])])

def align_2cad(infile='20160811_1005am.txt'):

    data = Table.read(infile, format='ascii')
    data['col4'] = data['col4'] * -1.0
    data['col6'] = data['col6'] * -1.0
    fid_upb = np.zeros(len(data), dtype='bool')
    fid_Sb = np.zeros(len(data), dtype='bool')
    fid_Nb = np.zeros(len(data), dtype='bool')
    aom2b = np.zeros(len(data), dtype='bool')
    dmb = np.zeros(len(data), dtype='bool')

    mm2in = 0.0393701 #mm in an inch

    for i in range(len(data)):
        if 'fid_up' in data['col2'][i]:
            fid_upb[i] = True
        if 'fid_S' in data['col2'][i]:
            fid_Sb[i] = True
        if 'fid_N' in data['col2'][i]:
            fid_Nb[i] = True
        if 'dm' in data['col2'][i]:
            dmb[i] = True
        if 'AOM2' in data['col2'][i] or 'aom2' in data['col2'][i]:
            aom2b[i] = True 

    fid_up = ave_sig_clip(data['col4'][fid_upb], data['col5'][fid_upb], data['col6'][fid_upb]) *mm2in
    print 'fid up', fid_up / mm2in
    fid_N = ave_sig_clip(data['col4'][fid_Nb], data['col5'][fid_Nb], data['col6'][fid_Nb])*mm2in
    print 'fid N', fid_N / mm2in
    fid_S = ave_sig_clip(data['col4'][fid_Sb], data['col5'][fid_Sb], data['col6'][fid_Sb])*mm2in
    print 'fid S', fid_S / mm2in
    aom2 = ave_sig_clip(data['col4'][aom2b], data['col5'][aom2b], data['col6'][aom2b])*mm2in
    print 'AOM2', aom2 / mm2in
    dm = ave_sig_clip(data['col4'][dmb], data['col5'][dmb], data['col6'][dmb])*mm2in
    print 'DM', dm / mm2in

    cad_coo = Table.read('cmm2model.txt', format='ascii')
    

    
    
    matfid = np.matrix([fid_up, fid_S, fid_N])
    cadmat = np.matrix([cad_coo['col2'], cad_coo['col3'], cad_coo['col4']]).T
    R, t = rigid_transform.rigid_transform_3D(matfid,cadmat )
    aom2new = np.dot(R, aom2) + t.T
    dmnew = np.dot(R, dm) + t.T
    fidnew = np.dot(R, matfid) +t.T

    #now transform all the points
    datmat = np.matrix([data['col4']*mm2in, data['col5']*mm2in, data['col6']*mm2in])
    import pdb;pdb.set_trace()
    coonew = np.dot(R, datmat) + t
    coonew = np.array(coonew) 
    _outtab = Table(data=[data['col1'], data['col2'], data['col3'], coonew[0,:], coonew[1,:], coonew[2,:]],names=['collection', 'group', 'name', 'x', 'y', 'z'])
    _outtab.write('transformed.txt', format='ascii.fixed_width')
    
    model = Table.read('model.txt', format='ascii')
    _dmb = model['col1'] == 'dm'
    _aom2 = model['col1'] == 'aom2'
    import pdb;pdb.set_trace()
    delta_dm = np.array(dmnew)[0] - np.array([model['col2'][_dmb], model['col3'][_dmb], model['col4'][_dmb]]).T
    delta_aom2 = np.array(aom2new)[0] - np.array([model['col2'][_aom2], model['col3'][_aom2], model['col4'][_aom2]]).T
    print 'differnece for dm, in inches ', delta_dm
    print 'difference for aom2, in inches ', delta_aom2

    
    return aom2new, dmnew, fidnew
    
    #just do it directly, because life is a mess
    #first get rid of centroids for both
    #matfid_cent = np.mean(matfid, axis=0)
    #cadmat_cent = np.mean(cadmat, axis=0)
    #matin = matfid - matfid_cent
    #cadmat_in = cadmat - cadmat_cent
    
    import pdb;pdb.set_trace()
    newcoo = np.array(R *mat1 + t)



def calc_angle(pcen , prcen, x, y, xr, yr):

    dcen = prcen - pcen
    xn = x + dcen[0]
    yn = y + dcen[1]

    import pdb;pdb.set_trace()
    R, t = rigid_transform.rigid_transform_3D(np.matrix([xn,yn, np.zeros(len(xn))]).T, np.matrix([xr,yr, np.zeros(len(xr))]).T, center=False)
    ang = np.rad2deg(np.arccos(R[0,0]))
    return ang

def bot_circ_ang(infile_cmm='top_plate_detailed.txt', infile_CAD='model_bushing.txt'):

    mm2in = 0.0393701
    data = Table.read(infile_cmm, format='ascii')
    bolt_circleB = data['col2'] == 'bolt_circle'
    xin = data['col4'][bolt_circleB] *mm2in
    yin = data['col5'][bolt_circleB] * mm2in
    bcenb = (data['col2'] == 'Main') * (data['col3'] == 'bolt_circle_center')
    cm_cen = np.array([data['col4'][bcenb], data['col5'][bcenb]]) *mm2in
    xin = xin - cm_cen[0]
    yin = yin - cm_cen[1]

    rin = np.sqrt(xin**2 + yin**2)
    theta = np.arctan2(yin , xin)

    model = Table.read(infile_CAD, format='ascii')
    rr = np.sqrt(model['col2']**2 + model['col3']**2)
    thetar = np.arctan2(model['col3'],model['col2'])

    #need to match the points
    indmod = []
    for i in range(len(model)):
        indmod.append(np.argmin(np.sqrt((xin - model['col2'][i])**2 + (yin - model['col3'][i])**2)))
        print np.min(np.sqrt((xin - model['col2'][i])**2 + (yin - model['col3'][i])**2))
    xin = xin[indmod]
    yin = yin[indmod]
    rin = rin[indmod]
    theta = theta[indmod]
    
    #import pdb;pdb.set_trace()
    #need cad model center
    #need cad coordiantes (in same order)

    #ang = calc_angle(cm_cen, cad_Cen, xin, yin




def alignAOM2top(botfile='AOM1_bottomount.txt', topfile='AOM1_topmount.txt'):

    data = Table.read(botfile, format='ascii')
    #data['col4'] = data['col4'] * -1.0
    #data['col6'] = data['col6'] * -1.0
    fid_mmb = np.zeros(len(data), dtype='bool')
    fid_mdb = np.zeros(len(data), dtype='bool')
    fid_mub = np.zeros(len(data), dtype='bool')
    fid_sb = np.zeros(len(data), dtype='bool')
    fid_nb = np.zeros(len(data), dtype='bool')

    fid_upb = np.zeros(len(data), dtype='bool')
    fid_Sb = np.zeros(len(data), dtype='bool')
    fid_Nb = np.zeros(len(data), dtype='bool')
    
    aom1Eb = np.zeros(len(data), dtype='bool')
    aom1Sb = np.zeros(len(data), dtype='bool')
    aom1Nb = np.zeros(len(data), dtype='bool')
    aom1_1b =  np.zeros(len(data), dtype='bool')
    aom1_2b =  np.zeros(len(data), dtype='bool')
    aom1_3b =  np.zeros(len(data), dtype='bool')
    
    #mm2in = 0.0393701 #mm in an inch
    mm2in = 1.0

    for i in range(len(data)):
        if 'fid_up' in data['col2'][i]:
            fid_upb[i] = True
        if 'fid_S' in data['col2'][i]:
            fid_Sb[i] = True
        if 'fid_N' in data['col2'][i]:
            fid_Nb[i] = True
        if 'nfid_mm' in data['col2'][i]:
            fid_mmb[i] = True
        if 'nfid_mu' in data['col2'][i]:
            fid_mub[i] = True
        if 'nfid_md' in data['col2'][i]:
            fid_mdb[i] = True 
        if 'nfid_n' in data['col2'][i]:
            fid_nb[i] = True
        if 'nfid_s' in data['col2'][i]:
            fid_sb[i] = True
        if 'aom1_S' in data['col2'][i]:
            aom1Sb[i] = True
        if 'aom1_E' in data['col2'][i]:
            aom1Eb[i] = True
        if 'aom1_N' in data['col2'][i]:
            aom1Nb[i] = True
        if 'AOM1_1' in data['col2'][i]:
            aom1Sb[i] = True
        if 'AOM1_2' in data['col2'][i]:
            aom1Eb[i] = True
        if 'AOM1_3' in data['col2'][i]:
            aom1Nb[i] = True

        
    #boollist = [fid_upb, fid_Sb, fid_Nb, fid_mmb, fid_mub, fid_mdb, fid_nb, fid_sb, aom1Sb, aom1Eb, aom1Nb]
    boollist = [fid_mmb, fid_mub, fid_mdb, fid_nb, fid_sb, aom1Sb, aom1Eb, aom1Nb]
    plist = []
    for indb in boollist:
        plist.append(ave_sig_clip(data['col4'][indb], data['col5'][indb], data['col6'][indb]) *mm2in)
    cad_coo = Table.read(topfile, format='ascii')

    #cad_coo['col4'] = cad_coo['col4'] * -1.0
    #cad_coo['col6'] = cad_coo['col6'] * -1.0

    Tfid_mmb = np.zeros(len(cad_coo), dtype='bool')
    Tfid_mdb = np.zeros(len(cad_coo), dtype='bool')
    Tfid_mub = np.zeros(len(cad_coo), dtype='bool')
    Tfid_sb = np.zeros(len(cad_coo), dtype='bool')
    Tfid_nb = np.zeros(len(cad_coo), dtype='bool')

    Tfid_upb = np.zeros(len(cad_coo), dtype='bool')
    Tfid_Sb = np.zeros(len(cad_coo), dtype='bool')
    Tfid_Nb = np.zeros(len(cad_coo), dtype='bool')
    for i in range(len(cad_coo)):
        if 'fid_up' in cad_coo['col2'][i]:
            Tfid_upb[i] = True
        if 'fid_S' in cad_coo['col2'][i]:
            Tfid_Sb[i] = True
        if 'fid_N' in cad_coo['col2'][i]:
            Tfid_Nb[i] = True
        if 'nfid_mm' in cad_coo['col2'][i]:
            Tfid_mmb[i] = True
        if 'nfid_mu' in cad_coo['col2'][i]:
            Tfid_mub[i] = True
        if 'nfid_md' in cad_coo['col2'][i]:
            Tfid_mdb[i] = True 
        if 'nfid_n' in cad_coo['col2'][i]:
            Tfid_nb[i] = True
        if 'nfid_s' in cad_coo['col2'][i]:
            Tfid_sb[i] = True
    #Tboollist = [Tfid_upb, Tfid_Sb, Tfid_Nb, Tfid_mmb, Tfid_mub, Tfid_mdb, Tfid_nb, Tfid_sb]
    Tboollist = [Tfid_mmb, Tfid_mub, Tfid_mdb, Tfid_nb]
    Tplist = []
    for indb in Tboollist:
        Tplist.append(ave_sig_clip(cad_coo['col4'][indb], cad_coo['col5'][indb], cad_coo['col6'][indb]) *mm2in)
    
    

    
    
    matfid = np.matrix(plist[:len(Tplist)])
    cadmat = np.matrix(Tplist)
    R, t = rigid_transform.rigid_transform_3D(matfid,cadmat )
    #aom2new = np.dot(R, aom2) + t.T
    #dmnew = np.dot(R, dm) + t.T
    #fidnew = np.dot(R, matfid.T) +t.T
    #print fidnew
    #now transform all the points
    datmat = np.matrix([data['col4']*mm2in, data['col5']*mm2in, data['col6']*mm2in])
    import pdb;pdb.set_trace()
    coonew = np.dot(R, datmat) + t
    coonew = np.array(coonew) 
    _outtab = Table(data=[data['col1'], data['col2'], data['col3'], coonew[0,:], coonew[1,:], coonew[2,:]], names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
    _outtab.write(botfile.replace('.txt', '')+'transformed.txt', format='ascii.fixed_width')
    
    model = Table.read('model.txt', format='ascii')
    _dmb = model['col1'] == 'dm'
    _aom2 = model['col1'] == 'aom2'
    import pdb;pdb.set_trace()
    delta_dm = np.array(dmnew)[0] - np.array([model['col2'][_dmb], model['col3'][_dmb], model['col4'][_dmb]]).T
    delta_aom2 = np.array(aom2new)[0] - np.array([model['col2'][_aom2], model['col3'][_aom2], model['col4'][_aom2]]).T
    print 'differnece for dm, in inches ', delta_dm
    print 'difference for aom2, in inches ', delta_aom2

    
    return aom2new, dmnew, fidnew
    
    #just do it directly, because life is a mess
    #first get rid of centroids for both
    #matfid_cent = np.mean(matfid, axis=0)
    #cadmat_cent = np.mean(cadmat, axis=0)
    #matin = matfid - matfid_cent
    #cadmat_in = cadmat - cadmat_cent
    
    import pdb;pdb.set_trace()
    newcoo = np.array(R *mat1 + t)



def diff_main_pts(f1='AOM1_topmount.txt', f2='topmount_new_fiducials_0823transformed.txt'):

    tab1 = Table.read(f1, format='ascii')
    tab2 = Table.read(f2, format='ascii.fixed_width')

    mbool = np.zeros(len(tab1),dtype='bool')
    for i in range(len(tab1)):
        if tab1['col2'][i] =='Main' or tab1['col2'][i] == 'bolt_circle':
            mbool[i] = True

    tab1 = tab1[mbool]
    mbool = np.zeros(len(tab2),dtype='bool')
    for i in range(len(tab2)):
        if tab2['col2'][i] =='Main' or tab2['col2'][i] == 'bolt_circle':
            mbool[i] = True

    tab2 = tab2[mbool]

    name = []
    dx = []
    dy = []
    dz = []
    dr = []
    for i in range(len(tab1)):
        match = False
        k = 0
        while not match:
            if tab1['col3'][i] == tab2['col3'][k]:
                name.append(tab1['col3'][i])
                dx.append(tab1['col4'][i] - tab2['col4'][k])
                dy.append(tab1['col5'][i] - tab2['col5'][k])
                dz.append(tab1['col6'][i] - tab2['col6'][k])
                dr.append(np.sqrt((tab1['col4'][i] - tab2['col4'][k])**2 + (tab1['col5'][i] - tab2['col5'][k])**2 + (tab1['col6'][i] - tab2['col6'][k])**2))
                match = True
            elif k == len(tab2)-1:
                match = True
            else:
                k+=1

    _out = Table(data=[name, dx, dy, dz, dr])
    _out.write('diff.txt', format='ascii.fixed_width')
    


def align_coo(lis_f, master_in=0):


    files = Table.read(lis_f, format='ascii.no_header')

    #first align all files to the master_in file

    #then average the coordinates of the new fiducials

    #then align all the files to the average fiducial measurements

    #then make a table of differences!


def reduce_cmm_file(file, form='ascii'):

    tab = Table.read(file, format=form)

    pgroups = np.unique(tab['col2'])
    pbool = np.ones(len(pgroups), dtype='bool')
    for  i, g in enumerate(pgroups):
        if g == 'bolt_circle' or g == 'Main' or 'tb' in g or g == 'telint' or g == 'aom2_mirror' or g== 'dm_mirror' or g == 'aom2_mount' or g == 'align_tel_cyl':
            pbool[i] = False
   
    pgroups = pgroups[pbool]

    x = []
    xerr = []
    y = []
    yerr = []
    z = []
    zerr = []
    
    for g in pgroups:
        _x = []
        _y = []
        _z = []
        print 'working though group ', g
        for i in range(len(tab)):
            
            if tab['col2'][i] == g:
                _x.append(tab['col4'][i])
                _y.append(tab['col5'][i])
                _z.append(tab['col6'][i])
                

        #import pdb;pdb.set_trace()
        _ave_x, _sig_x = statsIter.mean_std_clip(np.array(_x))
        _ave_y, _sig_y = statsIter.mean_std_clip(np.array(_y))
        _ave_z, _sig_z = statsIter.mean_std_clip(np.array(_z))

        x.append(_ave_x)
        xerr.append(_sig_x)
        y.append(_ave_y)
        yerr.append(_sig_y)
        z.append(_ave_z)
        zerr.append(_sig_z)
            
    _out = Table(data=[pgroups, x , xerr, y, yerr, z, zerr], names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'])
    _out.write('red_'+file, format='ascii.fixed_width')

            
            
def align_red(red1, red2):


    tab1 = Table.read(red1, format='ascii.fixed_width')
    tab2 = Table.read(red2, format='ascii.fixed_width')

    nfidb = np.zeros(len(tab1), dtype='bool')
    for i in range(len(tab1)):
        if 'nfid' in tab1['col1'][i]:
            nfidb[i] = True
    index2 = []
    for name in tab1['col1'][nfidb]:
        for i in range(len(tab2)):
            if name == tab2['col1'][i]:
                index2.append(i)
    mat1 = np.matrix([tab1['col2'][nfidb], tab1['col4'][nfidb], tab1['col6'][nfidb]]).T
    mat2 = np.matrix([tab2['col2'][index2], tab2['col4'][index2], tab2['col6'][index2]]).T
    R, t = rigid_transform.rigid_transform_3D(mat1, mat2)
    datmat = np.matrix([tab1['col2'], tab1['col4'], tab1['col6']])
    coonew = np.dot(R, datmat) + t
    coonew = np.array(coonew)

    _outtab = Table(data=[tab1['col1'], coonew[0,:], coonew[1,:], coonew[2,:]], names=['col1', 'col2', 'col3', 'col4'])
    _outtab.write(red1.replace('.txt', '')+'new_red_trans.txt', format='ascii.fixed_width')

    return tab1['col1'], coonew[0,:], coonew[1,:], coonew[2,:]


def test_align_red(fbase='red_topmount_v6.txt', trials=100,  err = 0.03, ang_max=3,  off=5):

    base = Table.read(fbase, format='ascii.fixed_width')
    b1 = Table.read(fbase, format='ascii.fixed_width')
    xb = base['col2']
    yb = base['col4']
    zb = base['col6']
    ave_delt = []
    sig_delt = []
    for i in range(trials):
        
        ang1 = np.deg2rad(np.random.uniform()*ang_max)
        ang2 = np.deg2rad(np.random.uniform()*ang_max)
        ang3 = np.deg2rad(np.random.uniform()*ang_max)

        R1 = np.matrix([[np.cos(ang1), np.sin(ang1)], [-np.sin(ang1), np.cos(ang1)]])
        coo = R1 * np.matrix([xb, yb])
        xn = np.array(coo)[0]
        yn = np.array(coo)[1]

        R2 = np.matrix([[np.cos(ang2), np.sin(ang2)], [-np.sin(ang2), np.cos(ang2)]])
        coo = R1 * np.matrix([xn, zb])
        xn =  np.array(coo)[0]
        zn = np.array(coo)[1]

        R3 = np.matrix([[np.cos(ang3), np.sin(ang3)], [-np.sin(ang3), np.cos(ang3)]])
        coo = R1 * np.matrix([yn, zn])
        yn =  np.array(coo)[0]
        zn = np.array(coo)[1]

        xn = xn + err * np.random.standard_normal(size=len(xn)) + np.random.uniform()*off 
        yn = yn + err * np.random.standard_normal(size=len(xn)) + np.random.uniform()*off
        zn = zn + err * np.random.standard_normal(size=len(xn)) + np.random.uniform()*off 
         
        
        b1['col2'] = xn
        b1['col4'] = yn
        b1['col6'] = zn
        b1.write('tmp.txt', format='ascii.fixed_width')

        name_new, xm, ym, zm = align_red('tmp.txt', fbase)

        dx = xm - xb
        dy = ym - yb
        dz = zm - zb
        dr = np.sqrt(dx**2 + dy**2 + dz**2)

        #import pdb;pdb.set_trace()
        ave_delt.append(np.max(dr))
        

    return ave_delt
        

        

        
