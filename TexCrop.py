import numpy as np
from astropy.io import fits

def TexCrop(Tex, CutoffValue, length, width):
    
    """
    !!! Need to reload original Tex every run (or else it uses the just cropped version)
    
    Function takes a Tex Map and Cutoff value and trims the Tex file to only values less than that cutoff.

    Parameters
    ---------------

    Tex: Fits File (2d)
        Excitation temperature map
        
    CutoffValue: (float)
        Limit at which to trim the file
        
    length: (int)
        Length of fits image in pixels
        
    width: (int)
        Width of fits image in pixels
        
    Returns
    ----------------
    Tex: Trimmed Tex map

    """
    
    for i in range(length):
        for j in range(width):
            if (Tex[0].data[i][j] > CutoffValue):
                Tex[0].data[i][j] = 0
    return Tex

