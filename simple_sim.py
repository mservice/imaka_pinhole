import numpy as np
import SFT
import matplotlib.pyplot as plt 

def mk_mask(coos, rads, ilum_lis):
    '''
    returns an array that is the complex amplitude at a pinhole mask

    arguments:
    coos must be positive, pixel coordinates of the pinhole location (reccomeded to use np.meshgrid to create, if regular pattern)
    rads -> list of corresponding radius of each pinhole
    ilum_list -> list corresponds to each pinhole.
    '''

    xcoo = coos[0].flatten()
    ycoo = coos[1].flatten()
    #xran = (np.max(xcoo) - np.min(xcoo)) + np.max(rads)
    #yran = (np.max(ycoo) - np.min(ycoo)) + np.max(rads)
    bx = int(np.max(xcoo) + np.max(rads))
    by = int(np.max(ycoo) + np.max(rads))
    if bx > by:
        bsize = bx
    else:
        bsize=by
    c_field = np.zeros((bsize,bsize ), dtype='complex')
    
    for i in range(len(xcoo)):
        c_temp = mk_pinhole(rads[i], ilum_lis[i])
        c_field[xcoo[i]-rads[i]:xcoo[i]+rads[i], ycoo[i]-rads[i]:ycoo[i]+rads[i]] = c_field[xcoo[i]-rads[i]:xcoo[i]+rads[i], ycoo[i]-rads[i]:ycoo[i]+rads[i]] + c_temp

    return c_field

def mk_pinhole(rad, ilum):
    i_ar = mk_ilum(ilum, rad)
    cpix = float(i_ar.shape[0])/2.0 - 0.5
    coo_ar = np.meshgrid(range(i_ar.shape[0]), range(i_ar.shape[1]))
    
    i_ar[((coo_ar[0]-(float(i_ar.shape[0])/2.0 -.5))**2 +(coo_ar[1]-(float(i_ar.shape[1])/2.0 -.5))**2) > rad**2] = 0 + 0j
    return i_ar  
    
def mk_ilum(arg, rad):

    if arg == 1:
        return np.ones((rad*2,rad*2), dtype='complex')




#def propogate(c_field, distance, nlamD):
#    '''
#    C_field is complex field at strting point
#    distance is distance oyou wish to propogate
#    nlamD is the size of the new complex field, in units of lamda/D
#    '''

#    t1 = 1 / np.sqrt( j * 

def diffract(c_field, d, lam, box_size, f_box_size, f_npix, quad=True):
    '''
    c_field is complex amplitude at the start.
    d is distance to output plane
    box_sizex is size of the c_field array in distance units (same as d and wavelgnth)
    lamh is the wavelegnth of light
    for now, makes final grid the same physical size as the input grid
    if quad, uses quadratic phase term (Fresnel Diffraction)
    else uses no quad phase term, which is Fraunhofer Diffraction
    '''
    coo = np.meshgrid(np.linspace(-box_size/2.0 , box_size/2.0, num=c_field.shape[0]), np.linspace(-box_size/2.0, box_size/2.0, num=c_field.shape[0]))
    coo_f = np.meshgrid(np.linspace(-f_box_size/2.0,f_box_size/2.0, num=f_npix), np.linspace(-f_box_size/2.0, f_box_size/2.0, num=f_npix))

    k = 2 * np.pi / lam
    t1 = np.exp( complex(0,1) * k * d) / (complex(0,1) * lam * d) * np.exp(complex(0,1)*k / (2 * d) * (coo_f[0]**2 + coo_f[1]**2))

    if quad:
        fft_arg = c_field * np.exp(complex(0,1) * k / (2 * d) * (coo[0]**2 + coo[1]**2))
        
    else:
        fft_arg = c_field
       
    nlamd = (f_box_size / d ) / (lam / box_size  )
    print nlamd
    fft_out = SFT.SFT3(fft_arg , nlamd, f_npix)
    return t1 * fft_out


    
def lens(c_field, focal_length, lam, box_size, f_box_size, f_npix):
    '''
    calulate the complex field at the focal plane of a lens, given the input c_field
    for a thin lens, this is the same as the FRAUNHFER Diffraction pattern.  Note that this is iplemented in diffract with quad = False
    '''

    c_out = diffract(c_field, focal_length, lam, box_size, f_box_size, f_npix, quad=False)
    return c_out

#def impulse_lens(lam, z1, z2, lens_mask, xobj, yobj,
#def im_lens_div(z1, z2, d, lam, 
def test_lens(plot=True):

    #define important qunatities
    #wavelegnth in meters
    lam = .75 * 10**-6
    #distance from pinhole mask to lens in meters
    d1 = .5
    #focal distance of the lens
    d2 = d1 / 2.
    #size of the orignal pinhole mask (meters)
    obj_size = .75*10**-3.0*10
    #size of plane in front of lens (meters)
    size_front_lens =  4*10**-3 
    #size on final focal plane
    size_focal_plane = 1.5 *10**-3
    #pinhole positions (pixels)
    hpix = [5,45,85,125,165,205,245,285,325,365,405,445,485]
    hpix = np.linspace(25, 475, num=10)
    hpix = np.array(hpix) 
    #hcoo = np.meshgrid([25, 75,125,25, 75, 125, 25, 75, 125  ],[25, 25,25, 75, 75, 75, 125, 125, 125]) # *10
    #hcoo = np.meshgrid([25],[25])
    hcoo - np.meshgrid(hpix, hpix)
    ilum = np.ones(hcoo[0].flatten().shape)
    #radius of holes in pixels
    rads = np.zeros(ilum.shape)+25
    #pixels in object plane

    #npix_obj = 490
    #pixels prior to lens
    npix_front_lens = 1000
    #pixels in final focal plane
    npix_focal_plane = 1000

    
    c_field = mk_mask(hcoo, rads, ilum)
    c_field_lens = diffract(c_field, d1 , lam, obj_size, size_front_lens, npix_front_lens, quad=True)
    c_field_focal = lens(c_field_lens, d2 , lam, size_front_lens, size_focal_plane, npix_focal_plane)
    intensity = np.abs(c_field_focal)**2
    
    if plot:
        plt.figure(1)
        plt.clf()
        plt.gray()
        plt.imshow(intensity)
        plt.colorbar()
        plt.title('Focal Plane Intensity')

        plt.figure(2)
        plt.clf()
        plt.imshow(np.log10(np.abs(c_field_lens)**2))
        plt.colorbar()
        plt.title('Log Intenity after pinholes diffraction')
        plt.show()

        plt.figure(3)
        plt.clf()
        plt.imshow(np.abs(c_field_lens)**2)
        plt.colorbar()
        plt.title('Intenity after pinholes diffraction')
        plt.show()

        plt.figure(4)
        plt.clf()
        plt.imshow(c_field.real)
        plt.colorbar()
        plt.title('Object')
       

    return c_field_focal
    
    


def im_sys(c_in, obj_coo, im_coo, z1, z2, lam=800*10**-9):
    '''
    uses direct application of Goodman 5-28 and 5-23 to calculate the image of a lens
    c_in is the complex field of the intensity
    obj_Coo are the coordinates associated with c_in
    im_coo are the pixel coorinates for the output?
    '''

    #for i in range(len(
    #integ = np.sum(np.exp(
    pass



    
def screen(z_dist, ap_size ,ap_samples, screen_size,screen_samples, lam, phase_map, mask_pixels, verb =False):
    #this uses PHASOR method of light propagation!!!!
    #z_dist is the distance from the aperture to the scren
    #screen_x is the x distance of each subsequent column.  It is  1-D vector
    #screen_y is y distance, also 1-D vector
    #due to input for screen_x, screen_y intrinsicaly assumes screen is rectangular
    #phase_map is an array that gives the initial phases for each point on the aperture
    #mask pixels is either 0 or one, same dimensions as phsse map, represents geometry of the aperture
    #lam is the wavelength of light, must have same units as the other distances


    #declares array to be the dimensions of the screen, tht is one value per sample point on aperture
    path_length = np.zeros(phase_map.shape)
    arrow_map = np.zeros((screen_samples,screen_samples))

    #this array is the brightness at each sampled point on the screen
    c_map = np.zeros(arrow_map.shape, dtype='complex')

    #ap_x_array = np.zeros((len(ap_x),len(ap_x)))
    coo = np.meshgrid(np.linspace(-ap_size/2.0 , ap_size/2.0, num=ap_samples), np.linspace(-ap_size/2.0, ap_size/2.0, num=ap_samples))
    ap_x_array = coo[0]
    ap_y_array = coo[1]

    coof = np.meshgrid(np.linspace(-screen_size/2.0 , screen_size/2.0, num=screen_samples), np.linspace(-screen_size/2.0, screen_size/2.0, num=screen_samples))
    screen_x = coof[0]
    screen_y = coof[1]
    
    #ap_y_array = np.zeros((len(ap_y),len(ap_y)))

    #for i in range(len(ap_x)):
    #    ap_x_array[i,:] = ap_x
    #for i in range(len(ap_y)):
    #    ap_y_array[:,i] = ap_y


    for x in range(len(screen_x)):
        if verb:
            print 'You are ',x, ' of ',len(screen_x) ,' done with this screen'
        for y in range(len(screen_y)):
            
            path_length = distance(screen_x[x,y],screen_y[x,y],ap_x_array,ap_y_array,z_dist) / lam
            #import pdb; pdb.set_trace()
            arrow_map = mask_pixels * np.exp(complex(0,1) * ((2 * np.pi * path_length)+phase_map)) 
            c_map[y,x] = np.sum(arrow_map)
                
                
            

           
    
    return c_map, arrow_map

def distance(xscreen,yscreen,x_ap,y_ap,z_dist = 0):
    #this function generates the path distances to the xscreen, yscreen on the screen
    #has an optional arguement for the aditional z_distance factor
    #x_ap and y_ap are the vectors containing the x,y positions of the 

    #dist = np.zeros((len(xscreen),len(yscreen)))

    dist = (1.0 * x_ap - xscreen)**2 + (1.0 * y_ap - 1.0 * yscreen)**2 + z_dist**2

    dist = dist**0.5

    return dist

def calc_phase_lens(samples,focal_length,lam,ap_side, tilt=0):
 

    #want to make a single lens with focal length
    #this means inducing a phase difference that creates a path length difference that causes everythin to line up at the focal length

    path_length_edge = (focal_length**2.0 + (ap_side[0])**2)**0.5
    phase = np.zeros((samples,samples))
    tilt_dist = focal_length * tilt
    
    for i in range(samples):
        for j in range(samples):
            dist = ((ap_side[i]-tilt_dist)**2 + ap_side[j]**2 + focal_length**2)**0.5
            path_diff = path_length_edge - dist 
            phase[j,i] = 2 * np.pi * path_diff / lam
            
    

    return phase

def diffract_pinholes(lam, coos, rads, d, obj_size, final_size, scut=50,  pix_size_ind=50.0, pix_size_whole=100.0):
    '''
    scale_fac is factor that sets the phyiscal size of the output *multiple of obj_size
    lam is wavelgnth pf light
    coos[0] are x cooridnates of poiholes, coos[1] are the y cooordinates of the pinholes (pixels), in coordiantes of the obj_size
    d is the distance fromt he pinhole mask that we are diffracting to
    pix_size_ind is the number
    obj_size is the size of A SINGLE PINHOLE ARRAY, that is it is the phusical size associated with the pixels length of pix_size_ind
    final size is PHYISCAL SIZE of final complex amplitude function
    '''
    xcoo = coos[0].flatten()
    ycoo = coos[1].flatten()

    pix_scale_final = (1.0*pix_size_whole) / final_size
    pix_final = int(pix_scale_final * obj_size * 4)
        

    #c_field_pinhole  = np.ones((pix_size_ind, pix_size_ind), dtype='complex')
    arcoo = np.meshgrid(range(pix_size_ind), range(pix_size_ind))
    r = np.sqrt((arcoo[0] - pix_size_ind/2.0)**2 + (arcoo[1] - pix_size_ind/2.0)**2)
    c_final = np.zeros((pix_final*1.5,pix_final*1.5), dtype='complex')
    int_final = np.zeros(c_final.shape)
    c_pin_dict = {}
    #import pdb;pdb.set_trace()
    for i in range(len(rads)):
        c_field_pinhole  = np.ones((pix_size_ind, pix_size_ind), dtype='complex')
        c_field_pinhole[r > rads[i]] = 0
        if not rads[i] in c_pin_dict.keys():
            c_pin = diffract(c_field_pinhole, d, lam, obj_size, c_final.shape[0]/pix_scale_final*2, c_final.shape[0]*2, quad=False)
            #take center of complex amplitude function
            #cutout = c_pin[(c_pin.shape[0] - scut) /2:(c_pin.shape[0] + scut) /2, (c_pin.shape[1] - scut) /2:(c_pin.shape[1] + scut) /2]
            c_pin_dict[rads[i]] = c_pin
        else:
            c_pin = c_pin_dict[rads[i]]

        box_size = c_final.shape[0]
        #import pdb;pdb.set_trace()

        
        xshift = xcoo[i] * obj_size  / pix_size_ind * pix_scale_final # / final_size #number of pixels to shift the cutout of the fft
        yshift = ycoo[i] * obj_size / pix_size_ind * pix_scale_final #/ final_size 
        print xshift, yshift
        print pix_scale_final
        c_final += c_pin[c_pin.shape[0]/2.0 - box_size/2.0 + xshift: c_pin.shape[0]/2.0 + box_size/2.0 +xshift , c_pin.shape[1]/2.0 - box_size/2.0 +yshift : c_pin.shape[1]/2.0 + box_size/2.0 + yshift]
        int_final += np.abs(c_pin[c_pin.shape[0]/2.0 - box_size/2.0 + xshift: c_pin.shape[0]/2.0 + box_size/2.0 +xshift , c_pin.shape[1]/2.0 - box_size/2.0 +yshift : c_pin.shape[1]/2.0 + box_size/2.0 + yshift])**2
        #import pdb; pdb.set_trace()
        
        
    plt.figure(26)
    plt.imshow(int_final)
    plt.show()
    #now take center slice of array
    #c_final_cent = c_final[c_final.shape[0]/2.0 - pix_size_whole/2.0:c_final.shape[0]/2.0 + pix_size_whole/2.0 ,c_final.shape[0]/2.0 - pix_size_whole/2.0:c_final.shape[0]/2.0 + pix_size_whole/2.0]
    import pdb;pdb.set_trace()
    return c_final
        
    

def test_phasors():


    #define important qunatities
    #wavelegnth in meters
    lam = .66 * 10**-6
    #distance from pinhole mask to lens in meters
    d1 = .46 * 2
    #focal distance of the lens
    d2 = .46
    #pitch of pinhole mask (meters)
    obj_size = 10**-2 #.50 * 10**-3
    #pixels between two pinholes
    obj_pixels = 100
    #size of plane in front of lens (meters)
    size_front_lens =  4950 * 10**-6
    #size on final focal plane
    size_focal_plane = 4950 * 10**-6
    sample_front_lens= 1000
    sample_focal_plane = 500
    #pinhole positions (pixels)
    #pinhole positions (pixels)
    #hpix = [5,45,85,125,165,205,245,285,325,365,405,445,485]
    hpix = np.linspace(-225, 225, num=10)
    hpix = np.array(hpix) 
    #hpix = [5,45,85,125,165,205,245,285,325,365,405,445,485]
    #hpix = np.array(hpix) 
    hcoo = np.meshgrid(hpix,hpix) 
    hcoo[0] = hcoo[0]  #* 2*10**-6
    hcoo[1] = hcoo[1]  #* 2*10**-6
    #hcoo = np.meshgrid([25],[25])

    ilum = np.ones(hcoo[0].flatten().shape)
    #radius of holes in pixels
    rads = np.zeros(ilum.shape)+.4 * obj_pixels
    #pixels in object plane

    #npix_obj = 490
    #pixels prior to lens
    
    #pixels in final focal plane
   


    #c_field = mk_mask(hcoo, rads, ilum)
    #import pdb; pdb.set_trace()
    #c_map, arrow_map = screen(d1, obj_size, c_field.shape[0], size_front_lens, sample_front_lens, lam, np.zeros(c_field.shape), c_field.real, verb=True)
    #import pdb; pdb.set_trace()
    c_field_lens = diffract_pinholes(lam, hcoo, rads, d1, obj_size,size_front_lens,  pix_size_ind=obj_pixels, pix_size_whole=sample_front_lens)
    #import pdb;psb.set_trace()
    #import pdb;pdb.set_trace()
    plt.figure(24)
    plt.imshow(np.abs(c_field_lens)**2)
    #c_field_lens = diffract_pinholes(c_field, d1 , lam, obj_size, size_front_lens, sample_front_lens, quad=True)
    #c_field_lens =c_map
    #import pdb; pdb.set_trace()
    phase_in = np.angle(c_field_lens)
    phase_lens = calc_phase_lens(c_field_lens.shape[0], d2, lam,  np.linspace(-size_front_lens/2.0, size_front_lens/2.0, num=c_field_lens.shape[0]))
    import pdb;pdb.set_trace()
    c_map_final, arrow_map = screen(d2*2, size_front_lens, c_field_lens.shape[0], size_focal_plane, sample_focal_plane, lam, phase_in + phase_lens, c_field_lens.real, verb=True)
    
    #c_field_lens = diffract(c_field, d1 , lam, obj_size, size_front_lens, npix_front_lens, quad=True)
    #c_field_focal = lens(c_field_lens, d2 , lam, size_front_lens, size_focal_plane, npix_focal_plane)
    #intensity = np.abs(c_field_focal)**2
 
       

    return c_map_final
