import pidly




def find(fits_file,thresh=100, fwhm=18 ):


    idl = pidly.IDL()
    idl('.compile find')
    idl('.compile fits_read')
    idl('fits_read, "'+fits_file+'", im, hdr')
    idl('find, im, x, y, flux, sharp, round, '+str(thresh)+','+str(fwhm)+', [0.1,1.0] , [-1.0,1.0]')
    import pdb;pdb.set_trace()
    return idl.x, idl.y, idl.flux
