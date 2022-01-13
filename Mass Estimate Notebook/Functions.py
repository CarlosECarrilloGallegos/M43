import numpy as np
from astropy.io import fits
from astropy import units as u
import math
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import pyregion
from astropy.wcs import WCS

#Load Fits Array
def load_fits(filename, *args):
    '''
    A function that takes in a FITS file and returns a python-useable array .
    
    Parameters
    ------------------
    
    filename: string
        name of the fits file
    *args: int
        Extension of file, if desired
    
    Returns
    ------------------
    (header, data)
        Tuple containing the header and data of the fits file

    '''
    
    
    fits_file = fits.open(filename)
    #header = fits_file[0].header
    #data = fits_file[0].data
    
    
    #return(header, data)
    return fits_file



#Integrated Intensity Map
def IIM (data, channel_min, channel_max, y_min, y_max, x_min, x_max):
    '''
    Function takes in a data cube, and parameters to slice it.
    Caclulates IIM (moment 0 map) using SpectralCube distribution.
    
    Parameters
    ---------------
    data: 3D Fits file (header included)
        Image cube to be sliced
    channel_min: float
        Lowest channel to include in the slicing
    channel_max: float
        Highest channel to include in the slicing
    y_min: float
        Lowest y value to include in the slciing
    y_max: float
        Highest y value to include in the slicing
    x_min: float
        Lowest x value to include in the slicing
    x_max: float
        Highest x value to include in the slicing
        
    Returns
    --------------
    IIM_map:
        Integrated intensity map of data cube
    
    '''
    #Make the data a spectral cube object
    cube = SpectralCube.read(data)
    
    #Slice it
    subcube = cube[channel_min:channel_max, y_min:y_max, x_min:x_max]
    
    #Apply .moment operator and divide by 1000 because Spectral Cube operates
    #in meters while we use km.
    
    IIM_map = subcube.moment(order = 0) / 1000
    
    return IIM_map

#Excitation Temperature

#Max Value


#Column Densities

def CD_Bourke_13CO(iim):
    '''
    Function takes in a 13CO fits cube and calculates its column density according to the
    methods in Bourke et. al 1997. 
    
    This approach assumes a constant excitation temperature.
    
    Parameters
    ------------------ 
    iim: 1D array
        Integrated intensity map of the region. 
        
    Returns
    -------------------
    CD_Map:
        Column density map of desired region
    '''
    #Define constant terms
    JTx = 47.28
    JTb = 0.818
    Tex = 50
    
    #Now calculate the leading term,a ssuming constant Tex = 50
    N_13CO = (2.42*10**14)*((Tex + 0.88)/(1-math.exp(-5.29/Tex)))*(1/(JTx - JTb))
    
    CD_Bourke_13CO = N_13CO * iim
    
    
    return CD_Bourke_13CO
def CD_Bourke_13CO_PinedaTau(iim, Tex, max_map):
    '''
    Function takes in a 13CO fits cube and calculates its column density according to the
    methods in Bourke et. al 1997. 
    
    This approach assumes a constant excitation temperature.
    
    Parameters
    ------------------ 
    iim: 1D array
        Integrated intensity map of the region. 
        
    Returns
    -------------------
    CD_Map:
        Column density map of desired region
    '''
    
    #Define Tau
    Tau = -np.log(1- (max_map / 5.3) / (1/(np.exp(5.3/Tex) -1) -0.16))
    
    #Now calculate the leading term
    N_13CO = (2.42*10**14)*((Tex + 0.88)/(1-math.exp(-5.29/Tex))) * Tau
    
    CD = N_13CO * iim
    
    
    return CD

def CD_Bourke_13CO_varTex(iim, Tex):
    '''
    Function takes in a 13CO fits cube and calculates its column density according to the
    methods in Bourke et. al 1997. 
    
    This approach assumes a constant excitation temperature.
    
    Parameters
    ------------------ 
    iim: 1D array
        Integrated intensity map of the region. 
        
    Returns
    -------------------
    CD_Map:
        Column density map of desired region
    '''
    #Define constant terms
    JTx = 47.28
    JTb = 0.818
    
    #Now calculate the leading term
    N_13CO = (2.42*10**14)*((Tex + 0.88)/(1-math.exp(-5.29/Tex)))*(1/(JTx - JTb))
    
    CD_Bourke_13CO = N_13CO * iim
    
    
    return CD_Bourke_13CO

def CD_Bourke_12CO_VarTex(iim, tex):
    '''
    Function takes in a 12CO fits cube and calculates its column density according
    to equation A10 in Bourke et. al 1997. 
    
    This approach uses a variable excitation temperature map.
    
    Parameters
    --------------
    iim: 2d array
        Integrated intensity map of 12CO data
    tex: 2d array
        Excitation temperature map of 12CO data
        
    Returns
    -------------
    CD_Bourke_12CO_VarTex:
        Column density map of 12CO
        
    '''
    
    #Define constants 
    h = 6.626 * 10**-34 #Planck's Constant
    v = 1.1527 * 10**11 #Frequency of 12CO 1-0
    k = 1.38 * 10**-23 #Boltzman constant
    Tb = 2.7 #Background temp
    
    # Evaluate Jtb, JTex
    Jtb = ((h*v)/k) * (1/np.exp((h*v)/(k*Tb))-1)
    JTex = ((h*v)/k) * (1/np.exp((h*v)/(k*(tex))-1))
    
    #Evaluate FTex
    FTex = (2.31 * 10**14)*((tex + 0.92)/(1 - np.exp(-5.53/tex))) * (1/(JTex - Jtb))
    
    #Compute Column Density
    CD_Bourke_12CO_VarTex = FTex * iim
    
    return CD_Bourke_12CO_VarTex

def CD_Pineda_unmasked(iim, Tex, max_map):
    '''
    Function returns column density map of data cube using the method in Kong et. al
    2019 (Pineda eqn 7/9).
    
    NOTE: We assume Tex of 12Co can be used for Tex 13CO as well.
    
    !!: currently set up for 13CO data only
    
    Parameters
    ---------------
    iim: 2d array
        Integrated intensity map of data
    tex: 2d array
        Excitation temperature map of data
    max_map: 2d array
        Map of maximum values of the data
    
    Returns
    --------------
    CD_Kong:
        Column Density map of data 
        
    '''
    
    
    #Evaluate Taue using Pineda equation 7
    Tau = -np.log(1- (max_map / 5.3) / (1/(np.exp(5.3/Tex) -1) -0.16))
    
    #Now Bourke eqn 5
    
    CD = ((Tau/(1 - np.exp(-Tau))) * (3*10**14) * (iim/(1-np.exp(-5.3/Tex))))
    
    return CD
def CD_Pineda_unmasked_opticthin(iim, Tex, max_map):
    '''
    Function returns column density map of data cube using the method in Kong et. al
    2019 (Pineda eqn 7/9).
    
    NOTE: We assume Tex of 12Co can be used for Tex 13CO as well.
    
    !!: currently set up for 13CO data only
    
    Parameters
    ---------------
    iim: 2d array
        Integrated intensity map of data
    tex: 2d array
        Excitation temperature map of data
    max_map: 2d array
        Map of maximum values of the data
    
    Returns
    --------------
    CD_Kong:
        Column Density map of data 
        
    '''
        
    #Evaluate Taue using Pineda equation 7
    Tau = -np.log(1- (max_map / 5.3) / (1/(np.exp(5.3/Tex) -1) -0.16))
    
    #Now Bourke eqn 5
    
    CD = (3*10**14) * (iim/(1-np.exp(-5.3/Tex)))
    
    return CD
def CD_Kong(iim, Tex, max_map):
    '''
    Function returns column density map of data cube using the method in Kong et. al
    2019 (Pineda eqn 7 followed by Bourke eqn 5).
    
    NOTE: We assume Tex of 12Co can be used for Tex 13CO as well.
    
    !!: currently set up for 13CO data only
    
    Parameters
    ---------------
    iim: 2d array
        Integrated intensity map of data
    tex: 2d array
        Excitation temperature map of data
    max_map: 2d array
        Map of maximum values of the data
    
    Returns
    --------------
    CD_Kong:
        Column Density map of data 
        
    '''
    
    #Use the Tex map to get a mask of the region
    Mask = np.where(Tex > 0, 1, 0)
    
    #Apply mask to Max value map
    Max_Masked = max_map * Mask
    
    #Evaluate Taue using Pineda equation 7
    Tau = -np.log(1- (Max_Masked / 5.3) / (1/(np.exp(5.3/Tex) -1) -0.16))
    
    #Now Bourke eqn 5
    
    CD_Kong = (2.42 * 10**14) * ((Tex + 0.88)/(1 - np.exp(-5.29/Tex))) * Tau
    
    return CD_Kong
def CD_Kong_constTEX(iim, max_map):
    '''
    Function returns column density map of data cube using the method in Kong et. al
    2019 (Pineda eqn 7 followed by Bourke eqn 5).
    
    NOTE: We assume Tex of 12Co can be used for Tex 13CO as well.
    
    !!: currently set up for 13CO data only
    
    Parameters
    ---------------
    iim: 2d array
        Integrated intensity map of data
    tex: 2d array
        Excitation temperature map of data
    max_map: 2d array
        Map of maximum values of the data
    
    Returns
    --------------
    CD_Kong:
        Column Density map of data 
        
    '''
    
    #Use the Tex map to get a mask of the region
    Tex = 50
    Mask = np.where(Tex > 0, 1, 0)
    
    #Apply mask to Max value map
    Max_Masked = max_map * Mask
    
    #Evaluate Taue using Pineda equation 7
    Tau = -np.log(1- (Max_Masked / 5.3) / (1/(np.exp(5.3/Tex) -1) -0.16))
    
    #Now Bourke eqn 5
    
    CD_Kong = (2.42 * 10**14) * ((Tex + 0.88)/(1 - np.exp(-5.29/Tex))) * Tau
    
    return CD_Kong

def add_header(CD):
    header = CD[0].header

    header.set('WCSAXES', 2)
    header.set('CRPIX1', 205.0)
    header.set('CRPIX2', -836.0)
    header.set('CDELT1', -0.000555555576728)
    header.set('CDELT2', 0.000555555576728)
    header.set('CUNIT1', 'deg')
    header.set('CUNIT2', 'deg')
    header.set('CTYPE1', 'RA---SIN')
    header.set('CTYPE2', 'DEC--SIN')
    header.set('CRVAL1', 83.9000372262)
    header.set('CRVAL2',-5.86858782674 )
    header.set('LONPOLE', 180.0)
    header.set('LATPOLE', -5.86858782674)
    header.set('RESTFRQ', 115271204000.0)
    header.set('RADESYS', 'FK5')
    header.set('EQUINOX', 2000.0)
    header.set('SPECSYS', 'LSRK')
    
    return CD
##################################

## MASS ESTIMATES BELOW ##

##################################

def Co12_Mass(CD, Region):
    #12CO mass estimate using Bourke eqn. A11
    #Required files: region file, 12CO CD with header
    
    #Variables
    
    H_12CO_Conversion = 10**(4)
    Area = 1.44*10**32 #cm^2
    u_m = 4.5*10**(-27) #kg

    #Load in and Mask the CD of 12CO
    region1 = Region #DS9 Region

    fdata = CD[0].data #column density values
    f = CD #fits file for column density

    #1
    region2 = region1.as_imagecoord(f[0].header)

    print (region2[0].coord_format)

    mymask = region2.get_mask(hdu=f[0])

    real_mask = mymask * fdata

    #fits.writeto("12CO_CD_Masked.fits", real_mask)

    CO_12_masked = real_mask

    #Compute the mass per pixel via eqn. A11
    Mass_per_pixel = H_12CO_Conversion * Area* u_m *CO_12_masked 

    #Sum all the values in the 2D array, that will give the mass
    Mass = np.sum(Mass_per_pixel)

    return Mass

def CO13_or_C18O_Mass(CD, Region):
    '''
    13CO mass estimate using Bourke eqn. A11
    
    Parameters
    -------------
    CD: 2d array
        Column Density Map
    Region: pyregion object
        Ds9 Region
        
    Returns
    --------------
    Mass: float
        kg mass of region
    '''
    #Variables
    
    H_13CO_Conversion = 7e5
    Area = 1.44*10**32 #cm^2
    #Make distance variable ^^
    u_m = 4.5*10**(-27) #kg

    #Load in and Mask the CD of 12CO
    region1 = Region

    fdata = CD[0].data
    f = CD

    #1
    region2 = region1.as_imagecoord(f[0].header)

    print (region2[0].coord_format)

    mymask = region2.get_mask(hdu=f[0])

    real_mask = mymask * fdata

    #fits.writeto("13CO_CD_Masked.fits", real_mask)

    CO_13_masked = real_mask

    #Compute the mass per pixel via eqn. A11
    Mass_per_pixel = H_13CO_Conversion * Area* u_m *CO_13_masked 

    #Sum all the values in the 2D array, that will give the mass
    Mass = np.sum(Mass_per_pixel)

    return Mass

