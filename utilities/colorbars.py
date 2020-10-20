#!/usr/bin/python

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class cbar(object):
    """
        Base class for color bars
    """

    def __init__(self):
        self.cbar = None

    def make_cbar(self, name, RGB_list):
        self.cbar = LinearSegmentedColormap.from_list(name, RGB_list)
        plt.register_cmap(cmap=self.cbar)

    def __call__(self, val):
        return self.cbar(val)


class cring(object):
    """
        Base class for periodic color bars
    """

    def __init__(self):
        self.cbar = None

    def make_cbar(self, name, RGB_list):
        self.cbar = LinearSegmentedColormap.from_list(name, RGB_list)
        plt.register_cmap(cmap=self.cbar)

    def __call__(self, val):
        return self.cbar(val)


class cbarHot(cbar):
    """
        Class to make the custom 'cbarHot' COSMO colorbar
    """

    def __init__(self, mn=0.0, mx=1.0):

        # 5th degree polynomial fit to the individual RGB values of the
        # Mathematica COSMO colorbar at 1000 sample points
        x = np.linspace(mn, mx, 100)

        # R, G, B value polynomial coefficients in decreasing order
        r = np.array([28.384, -66.085, 54.494, -16.557, -0.232, 0.991])
        g = np.array([4.196, -8.275, 9.143, -5.479, -0.086, 0.995])
        b = np.array([-2.312, 4.468, -3.085, -0.128, 0.052, 0.997])

        # Build polynomial
        pr = np.polyval(r, x)
        pg = np.polyval(g, x)
        pb = np.polyval(b, x)

        # Wrap back RGB values into the [0, 1] allowed range
        # (endpoints may fall outside [0, 1] due to error in the fit)
        for i in [pr, pg, pb]:
            i[np.where(i > 1.0)] = 1.0
            i[np.where(i < 0.0)] = 0.0

        # Build colormap and register it with Matplotlib
        cbar_name = 'cbarHot'
        if(mn!=0 or mx!=1.0):
            cbar_name +='_{}_{}'.format(mn, mx)
        self.make_cbar(cbar_name, np.column_stack((pr, pg, pb)))


class cbarHot_alt(cbar):
    """
        Class to make the custom 'cbarHot' COSMO colorbar in an alternate
        construction, based on blending with an existing Python colorbar.
        Appearance is not quite the same as the Mathematica colorbar,
        but is similar.
    """

    def __init__(self, mn=0.0, mx=1.0):

        # Build color bar on 1000 sample points
        x = np.linspace(mn, mx, 1000)

        # Extract 1000 sample points from existing color bar
        # Only use partial segment of the colorbar (0.0 to 0.8)
        y = np.linspace(0.0, 0.8, 1000)

        # RGB values
        rgbWhite = np.array([1.0, 1.0, 1.0])*np.ones((1000, 3))
        rgb1 = np.array([1.0, 0.5, 0.0])*np.ones((1000, 3))
        rgbPurple = np.array([0.5, 0.0, 0.5])*np.ones((1000, 3))

        # Blend with the 'ocean' colorbar
        cm = plt.get_cmap('ocean')

        # Blend ratios
        dsc = cm(1.0-y)**1.1
        c1 = (1.0-x)**4
        c2 = x**10
        c3 = (x-0.1)**2

        # Blend colors
        blend1 = np.einsum('ij,i->ij', dsc[:, 0:3], 1.0-c1) \
            + np.einsum('ij,i->ij', rgbWhite, c1)
        blend2 = np.einsum('ij,i->ij', blend1, 1.0-c3) \
            + np.einsum('ij,i->ij', rgbPurple, c3)
        blend3 = np.einsum('ij,i->ij', blend2, 1.0-c2) \
            + np.einsum('ij,i->ij', rgb1, c2)

        # Build the colormap and register it with Matplotlib
        cbar_name = 'cbarHot'
        if(mn!=0 or mx!=1.0):
            cbar_name +='_{}_{}'.format(mn, mx)
        self.make_cbar(cbar_name, blend3)


class cbarBWR(cbar):
    """
        Class to make the custom 'cbarBWR' COSMO colorbar.
        A similar colorbar exists already in Matplotlib, and
        (in my opinion) looks a little nicer: 'RdBu'
    """

    def __init__(self, mn=0.0, mx=1.0):

        # Build colorbar using 1000 sample points
        x = np.linspace(mn, mx, 1000)

        # RGB values
        rgbBlack = np.array([0.0, 0.0, 0.0])*np.ones((1000, 3))
        rgbWhite = np.array([1.0, 1.0, 1.0])*np.ones((1000, 3))
        rgb1 = np.array([0.0, 0.4, 1.0])*np.ones((1000, 3))
        rgb2 = np.array([1.0, 0.3, 0.0])*np.ones((1000, 3))

        # Blend ratios
        c1 = np.sin(x*np.pi/2)
        c2 = np.cos(x*np.pi/2)
        c3 = 2*np.sin(x*np.pi)**4 + 4*np.sin(x*np.pi)**16 \
            + 8*np.sin(x*np.pi)**64 + 32*np.sin(x*np.pi)**256
        c3 /= np.amax(c3)
        c4 = np.cos(x*np.pi)**16

        # Normalize ratios to add to 1
        norm = np.amax(c1+c2+c3+c4)
        c1 /= norm
        c2 /= norm
        c3 /= norm
        c4 /= norm

        # Blend colors
        blend = np.einsum('ij,i->ij', rgb2, c1) \
            + np.einsum('ij,i->ij', rgb1, c2) \
            + np.einsum('ij,i->ij', rgbWhite, c3) \
            + np.einsum('ij,i->ij', rgbBlack, c4)

        # Build the colormap and register it with Matplotlib
        cbar_name = 'cbarBWR'
        if(mn!=0 or mx!=1.0):
            cbar_name +='_{}_{}'.format(mn, mx)

        self.make_cbar(cbar_name, blend)


class cbarPhi(cring):
    """
        Class to make custom COSMO periodic colorbar
    """

    def __init__(self, mn=0.0, mx=1.0):

        # Build colorbar using 1000 sample points
        x = np.linspace(mn, mx, 1000)

        # RGB values
        rgbBlack = np.array([0.0, 0.0, 0.0])*np.ones((1000, 3))
        rgbWhite = np.array([1.0, 1.0, 1.0])*np.ones((1000, 3))
        rgb1 = np.array([0.0, 0.4, 0.9])*np.ones((1000, 3))
        rgb2 = np.array([1.0, 0.3, 0.0])*np.ones((1000, 3))

        # Blend ratios
        c1 = 0.5*np.cos(x*np.pi+np.pi/4+np.pi/4)**4
        c2 = 1.0*np.sin(x*np.pi+np.pi/4+np.pi/4)**8
        c3 = np.cos(x*np.pi+np.pi/4)**4
        c4 = np.cos(x*np.pi+np.pi/2+np.pi/4)**4

        # Normalize ratios to sum to 1
        norm = np.amax(c1+c2+c3+c4)
        c1 /= norm
        c2 /= norm
        c3 /= norm
        c4 /= norm

        # Blend colors
        blend = np.einsum('ij,i->ij', rgbBlack, c1) \
            + np.einsum('ij,i->ij', rgbWhite, c2) \
            + np.einsum('ij,i->ij', rgb1, c3) + \
            np.einsum('ij,i->ij', rgb2, c4)

        # Build the colormap and register it with Matplotlib
        cbar_name = 'cbarPhi'
        if(mn!=0 or mx!=1.0):
            cbar_name +='_{}_{}'.format(mn, mx)

        self.make_cbar(cbar_name, blend)


class cbarMPLtrunc(cbar):

    def __init__(self, mpl_cmap='jet', mn=0.0, mx=1.0):

        # Build colorbar using 1000 sample points
        x = np.linspace(mn, mx, 1000)
        cmap = mpl.cm.get_cmap(mpl_cmap)

        blend = np.array([cmap(xx) for xx in x])

        # Build the colormap and register it with Matplotlib
        cbar_name = mpl_cmap
        if(mn!=0 or mx!=1.0):
            cbar_name +='_{}_{}'.format(mn, mx)

            self.make_cbar(cbar_name, blend)

def load():
    for cb in [cbarHot, cbarBWR, cbarPhi]:
        cb(mn=0.0, mx=1.0)
    cbarHot(mn=0.3, mx=1.05)
    cbarMPLtrunc('bone_r',0.2, 1.0)
    cbarMPLtrunc('Reds',0.3, 1.0)
    cbarMPLtrunc('Blues',0.3, 1.0)
