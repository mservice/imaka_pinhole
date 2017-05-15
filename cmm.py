import numpy as np
from astropy.table import Table
from . import rigid_transform, fit_cyl
import matplotlib.pyplot as plt

def find_trans(f1, f2, outfile='trans.txt'):
    '''
    finds rotation and translation between two data sets, using all data points
    '''

    tab1 = Table.read(f1, format='ascii')
    tab2 = Table.read(f2, format='ascii')
    #need to match based on name first
    

    #need to turn coordiantes into matrix with correct form
    #tab1/2 shoudl have column named 'use' that is 0 or 1, if 0 then that coordiante shoudl not be used to derive the tranformation
    #totbool = (tab1['use'] * tab2['use']) == 0
    #totbool = np.ones(
    ind1 ,ind2 = match_on_name(tab1['col1'] , tab2['col1'])
    #import pdb;pdb.set_trace()
    mat1 = np.matrix([tab1['col2'][ind1], tab1['col3'][ind1], tab1['col4'][ind1]]).T
    mat2 =  np.matrix([tab2['col2'][ind2], tab2['col3'][ind2], tab2['col4'][ind2]]).T
    print 'using ', len(ind1), '  points'

    R, t = rigid_transform.rigid_transform_3D(mat1, mat2)


    mat1 = np.matrix([tab1['col2'], tab1['col3'], tab1['col4']])
    newcoo = R * mat1 + t

    xn = newcoo[0]
    yn = newcoo[1]
    zn = newcoo[2]
    xnn = []
    ynn = []
    znn = []
    
    for i in range(xn.shape[1]):
        xnn.append(xn[0,i])
        ynn.append(yn[0,i])
        znn.append(zn[0,i])
        
        
    outtab = Table(data=[xnn, ynn, znn, tab1['col1']], names=['x', 'y', 'z', 'name'])
    outtab.write(outfile, format='ascii.commented_header')

    

def match_on_name(l1, l2):

    ind1 = []
    ind2 = []
    for i in range(len(l1)):
        match = False
        kk = 0
        while not match:
            if l1[i] == l2[kk]:
                ind1.append(i)
                ind2.append(kk)
                match = True
            elif kk == len(l2)-1:
                match = True
            kk+=1

    return ind1, ind2


def align_all(lis_f, ref_f):
    lis_lis = Table.read(lis_f, format='ascii.no_header')['col1']

    for i in range(len(lis_lis)):
        find_trans(lis_lis[i], ref_f, outfile='trans'+str(i)+'.txt')


def collate(lis_f, outfile='collated.txt'):
    lis_lis = Table.read(lis_f, format='ascii.no_header')['col1']


    names = []
    for i in range(len(lis_lis)):
        _tmp = Table.read(lis_lis[i],format='ascii')
        for j in range(len(_tmp)):
            names.append(_tmp['name'][j])

    names = np.array(names)
    name_all = np.unique(names)
    ndict = {}
    for i in range(len(name_all)):
        ndict[name_all[i]] = i
        
    posall = np.ma.array(np.zeros((3 , len(lis_lis), len(name_all))))
    posall.mask = np.ones(posall.shape, dtype='bool')

    for i in range(len(lis_lis)):
        _tmp = Table.read(lis_lis[i],format='ascii')
        for j in range(len(_tmp)):
            posall[0,i,ndict[_tmp['name'][j]]] = _tmp['x'][j]
            posall.mask[0,i,ndict[_tmp['name'][j]]] = False
            posall[1,i,ndict[_tmp['name'][j]]] = _tmp['y'][j]
            posall.mask[1,i,ndict[_tmp['name'][j]]] = False
            posall[2,i,ndict[_tmp['name'][j]]] = _tmp['z'][j]
            posall.mask[2,i,ndict[_tmp['name'][j]]] = False


    pos_ave = np.mean(posall, axis=1)
    pos_err = np.std(posall, axis=1)
    N = posall.shape[1] - np.sum(posall.mask, axis=1)[0]

    #write outfile
    _out = Table(data=[name_all, pos_ave[0,:], pos_err[0,:], pos_ave[1,:], pos_err[1,:], pos_ave[2,:], pos_err[2,:],N ], names=['name', 'x', 'xerr', 'y', 'yerr', 'z', 'zerr', 'N'])
    _out.write(outfile, format='ascii.fixed_width')
    import pdb;pdb.set_trace()


def plot_trans_ref(trans_f, ref_f):

    trans_lis = Table.read(trans_f, format='ascii.no_header')['col1']
    for i in range(len(trans_lis)):
        _tmp = Table.read(trans_lis[i], format='ascii.commented_header')
        plt.figure(i)
        plt.clf()
        plt.scatter(_tmp['x'], _tmp['z'])
        for kk, txt in enumerate(_tmp['name']):
            plt.annotate(txt, (_tmp['x'][kk], _tmp['z'][kk]))

    ref = Table.read(ref_f, format='ascii')
    plt.figure(i+1)
    plt.clf()
    plt.scatter(ref['col2'], ref['col4'])
    for kk, txt in enumerate(ref['col1']):
        plt.annotate(txt, (ref['col2'][kk], ref['col4'][kk]))
        



def mk_resid(collate_file='collated.txt', ref_f='StructureCoordinates.txt'):
    '''
    '''
    ref = Table.read(ref_f, format='ascii')
    data = Table.read(collate_file, format='ascii.fixed_width')
    

    ind1, ind2 = match_on_name(data['name'], ref['col1'])

    names = data['name'][ind1]
    dx = data['x'][ind1] - ref['col2'][ind2]
    dy = data['y'][ind1] - ref['col3'][ind2]
    dz = data['z'][ind1] - ref['col4'][ind2]

    

    _out = Table(data=[names,  ref['col2'][ind2], dx,  ref['col3'][ind2], dy, ref['col4'][ind2], dz], names=['name', 'x', 'dx', 'y', 'dy', 'z', 'dz'])
    _out.write('residual.txt', format='ascii.fixed_width')



def fit_data(txt_f):

    '''
    '''

    #first need to do selection of points with correct location

    txt_tab = Table.read(txt_f, format='ascii')
    plane_bool = np.zeros(len(txt_tab), dtype='bool')

    for i in range(len(txt_tab)):
        if 'plane' in txt_tab['col1'][i]:
            plane_bool[i] = True

    plane_name = np.unique(txt_tab['col1'][plane_bool])
    cyl_name = []
    for i in range(len(plane_name)):
        cyl_name.append(plane_name[i].replace('plane', 'cyl'))

    pname = []
    p = []
    for i in range(len(plane_name)):
        _pbool = (txt_tab['col1'] == plane_name[i])
        _cbool = (txt_tab['col1'] == cyl_name[i])
        _p = fit_cyl.fit_point(txt_tab[_cbool], txt_tab[_pbool])
        p.append(_p)
        pname.append(plane_name[i].replace('plane',''))

    return pname, p
