#from __future__ import division

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double exp(double)
    double sqrt(double)
    double M_PI

# import g_math
import units
import numpy

__all__ = ['Physiology']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

cdef class Physiology(object):
    """This class implements physiological parameters and functions related to 
    blood flow.
    """

    cdef public dict _sf
    
    def __init__(self, defaultUnits={'length': 'um', 'mass': 'ug', 
                                     'time': 'ms'}):
        """Initializes the Physiology object.
        INPUT: defaultUnits: The default units to be used for input and output
                             as a dictionary, e.g.: {'length': 'm', 
                             'mass': 'kg', 'time': 's'}
        OUTPUT: None
        """
        
        self.tune_to_default_units(defaultUnits)
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    def tune_to_default_units(self, defaultUnits):
        """Tunes the Physiology object to a set of default units. This results
        in faster execution time and less calling parameters.
        INPUT: defaultUnits: The default units to be used for input and output
                             as a dictionary, e.g.: {'length': 'm', 
                             'mass': 'kg', 'time': 's'}
        OUTPUT: None
        """
        self._sf = {}
        sf = self._sf
        sf['um -> du'] = units.scaling_factor_du('um', defaultUnits)
        sf['mm/s -> du'] = units.scaling_factor_du('mm/s', defaultUnits)
        sf['kg/m^3 -> du'] = units.scaling_factor_du('kg/m^3', defaultUnits)
        sf['mmHg -> du'] = units.scaling_factor_du('mmHg', defaultUnits)
        sf['Pa*s -> du'] = units.scaling_factor_du('Pa*s', defaultUnits)
        sf['fL -> du'] = units.scaling_factor_du('fL', defaultUnits)
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef discharge_to_tube_hematocrit(self, double discharge_ht, double d,
                                       bint invivo):
        """Converts discharge hematocrit to tube hematocrit based on the
        formulas provided by Pries 2005 ('Microvascular
        blood viscosity in vivo and the endothelial surface layer')
        It can be chosen from an invivo and an invitro formulation.
        INPUT: discharge_ht: Discharge Ht expressed as a fraction [0,1]
               d: Diameter of the vessel (in microns)
               invivo: Boolean, whether or not to consider ESL influence.
        OUTPUT: tube_hematocrit: Tube hematocrit expressed as a fraction
                [0,1]
        """
        
        cdef double htd, dph, htt, x

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.    
        d = d / self._sf['um -> du']
        htd = discharge_ht

        if invivo:
            dph = self.physical_vessel_diameter(d)
            x = 1 + 1.7 * exp(-0.415*dph) - 0.6 * exp(-0.011*dph)
            htt = htd**2 + htd * (1-htd) * x
            return htt / (d / dph)**2
            # Note that the above differs from the (likely incorrect) formula in Pries
            # et al. 2005, which would result in: return htd * (d / dph)**2
        else:
            x = 1 + 1.7 * exp(-0.415*d) - 0.6 * exp(-0.011*d)
            htt = htd**2 + htd * (1-htd) * x
            return htt

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef tube_to_discharge_hematocrit(self, double tube_ht, double d, 
                                       bint invivo):
        """Converts tube hematocrit to discharge hematocrit based on the
        formulas provided by Pries 2005 ('Microvascular
        blood viscosity in vivo and the endothelial surface layer')
        It can be chosen from an invivo and an invitro formulation.
        NOTE 1: If the diameter is > 1000 micrometers the in vitro formulation 
        does not work and is not physiological. The discharge hematocrit is set equal to the tube hematocrit. 
        NOTE 2: In extreme cases in both formulations htd > 1.0 can be obtained. htd is always 
        bounded to 1.0.
        INPUT: tube_ht: Discharge Ht expressed as a fraction [0,1]
               d: Diameter of the vessel (in microns)
               invivo: Boolean, whether or not to consider ESL influence.
        OUTPUT: discharge_hematocrit: Discharge Ht expressed as a fraction
                [0,1]
        """
        
        cdef double dph, htd, x

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.    
        d = d / self._sf['um -> du']
        
        if invivo:
            dph = self.physical_vessel_diameter(d)
            x = 1 + 1.7 * exp(-0.415*dph) - 0.6 * exp(-0.011*dph)
            htt=tube_ht*(d / dph)**2
            htd = 0.5*(x - sqrt(-4*htt*x + x**2 + 4*htt))/(x - 1)
            if htd > 0.99:
                htd = 1.0
            return htd
            # Note that the above differs from the (likely incorrect) formula in Pries
            # et al. 2005, which would result in: return htd / (d / dph)**2
        else:
            x = 1 + 1.7 * exp(-0.415*d) - 0.6 * exp(-0.011*d)
            if d < 1000:
                htd = 0.5*(x - sqrt(-4*tube_ht*x + x**2 + 4*tube_ht))/(x - 1)
            else:
                htd=tube_ht
            if htd > 0.99:
                htd = 1.0
            return htd

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    cpdef relative_apparent_blood_viscosity(self, double diameter, 
                                            double discharge_hematocrit,
                                            bint invivo):
        """Returns the relative apparent blood-viscosity value, according to
        the given vessel specifications and discharge hematocrit.

        'Apparent', because the viscosity of blood as a suspension is not an
        intrinsic fluid property, but depends on the vessel size.
        'Relative', because the viscosity is scaled to (divided by) the
        viscosity of the solvent, i.e. the viscosity of blood plasma (as such,
        it should also be independent of temperature, Barbee 1973).
        Fit function taken from Pries 1992 (Blood viscosity in tube flow:
        dependence on diameter and hematocrit). 
        INPUT: diameter: Vessel diameter (in microns)
               discharge_hematocrit: Discharge Ht expressed as a fraction [0,1]
               invivo: This boolean determines whether to compute in vivo
                       viscosity or in vitro.
        OUTPUT: Relative apparent blood viscosity [1.0]
        """

        cdef double d, ht_d, nu45, c, nu, deff
        
        #if discharge_hematocrit=1 we use 0.99 for the calculation, otherwise the function yields inf)
        if discharge_hematocrit == 1.0:
            discharge_hematocrit = 0.99
                                   
        # The fit function assumes diameters in [micron]. Conversion from 
        # default units is performed as required.
        sf = self._sf['um -> du']
        
        if invivo:
            d = self.physical_vessel_diameter(diameter) / sf
        else:
            d = diameter / sf

        ht_d = discharge_hematocrit
        # relative apparent viscosity at Ht = 0.45:
        nu45 = 220.0 * exp(-1.3*d) + 3.2 - 2.44 * exp(-0.06*d**0.645)
        # exponent c:
        c = (0.8 + exp(-0.075*d)) * \
            (-1.0 + 1.0/(1.0 + 10.0**-11.0 * d**12.0)) + \
            1.0/(1.0 + 10.0**-11.0 * d**12.0)
        # relative apparent viscosity:           
        nu = 1.0 + (nu45 - 1.0) * ((1.0 - ht_d)**c-1.0) / \
                    ((1.0 - 0.45)**c-1.0)
        if invivo:
            deff = self.effective_vessel_diameter(diameter, 
                                                  discharge_hematocrit)
            nu = nu * (diameter/deff)**4.

        return nu
       
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef dynamic_plasma_viscosity(self,str plasmaType='default'):
        """Returns the dynamic viscosity of human plasma at 37 degrees 
        centigrate, as reported in 'Plasma viscosity: A forgotten variable' by 
        Kesmarky et al, 2008.
        INPUT: None
        OUTPUT: Dynamic viscosity of human plasma.
        """
        
        # The value reported by Kesmarky is scaled from [Pa s] to default 
        # units:
        if plasmaType == 'default':
            return 0.0012 * self._sf['Pa*s -> du']
        elif plasmaType == 'human':
            return 0.001339 * self._sf['Pa*s -> du']
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef rbc_volume(self, str species='rat'):
        """Returns the volume of a red blood cell (RBC)
        The red blood cell volumes differ between species. According to
        Baskurt and coworkers (1997), the respective value for human and rat
        erythrocytes is 89.16 and 56.51 fl respectively.
        Note: a more recent work by Windberger et al. (2003) lists
        considerably lower values.
        INPUT: None.
        OUTPUT: The volume of an RBC. 
        """

        if species == 'rat':
            return 56.51 * self._sf['fL -> du']
        elif species == 'human':
            return 89.16 * self._sf['fL -> du']
        elif species == 'mouse':
            return 49.0 * self._sf['fL -> du']
        elif species == 'human2':
            return 92.0 * self._sf['fL -> du']

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
        
    cpdef velocity_factor(self, double diameter, bint invivo, 
                          double discharge_ht=-1.0, double tube_ht=-1.0):
        """Returns the factor by which red blood cells (RBCs) are faster than
        the mean velocity.
        Hd = V Ht v F / (V v)  <==> F = Hd / Ht
        This function makes use of the empirical data provided / compiled by
        Pries and coworkers. Either tube or discharge hematocrit should be provided.
        INPUT: diameter: The diameter of the containing vessel.
               invivo: Boolean, whether or not to consider ESL influence.
               tube_ht: Tube hematocrit. 
               discharge_ht: Discharge hematocrit. 
        OUTPUT: The factor by which the RBC speed exceeds the mean blood
                velocity.
        """
        cdef double htd, htt

        if discharge_ht != -1.0:
            htd = discharge_ht
            htt = self.discharge_to_tube_hematocrit(htd, diameter, invivo)
        elif tube_ht != -1.0:
            htt = tube_ht
            htd = self.tube_to_discharge_hematocrit(htt, diameter, invivo)

        return htd/htt

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef physical_vessel_diameter(self, double diameter):
        """Returns the physical vessel diameter, which is the anatomical 
        diameter minus twice the physical width of the endothelial surface 
        layer (which is hematocrit independent), according to the empirical 
        model of Pries and Secomb in their 2005 paper 'Microvascular blood 
        viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
        OUTPUT: weff: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double wph = self.physical_esl_thickness(diameter)
        
        return diameter - 2*wph

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef physical_esl_thickness(self, double diameter):
        """Returns the physical width of the endothelial surface layer (which 
        is independent of hematocrit), according to the empirical model of 
        Pries and Secomb in their 2005 paper 'Microvascular blood viscosity in 
        vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
        OUTPUT: Physical width of the ESL.
        """
        
        cdef double sf, doff, dcrit, d50, eamp, ewidth, epeak, wmax

        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        diameter = diameter / sf
        
        # Parameters optimized for a hematocrit dependent ESL-impact on flow:
        doff = 2.4
        dcrit = 10.5
        d50 = 100.
        eamp = 1.1
        ewidth = 0.03
        epeak = 0.6
        wmax = 2.6

        if diameter <= doff:
            was = 0.
            wpeak = 0.
        elif diameter <= dcrit:
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
            wpeak = eamp * (diameter-doff) / (dcrit-doff)
        else:
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
            wpeak = eamp * exp(-ewidth*(diameter - dcrit))
        
        return (was + wpeak * epeak) * sf
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef effective_vessel_diameter(self, double diameter, double htd):
        """Returns the effective vessel diameter, which is the anatomical 
        diameter minus twice the effective width of the endothelial surface 
        layer (which depends on the discharge hematocrit), according to the 
        empirical model of Pries and Secomb in their 2005 paper 'Microvascular 
        blood viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
               htd: Discharge hematocrit.
        OUTPUT: weff: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double weff = self.effective_esl_thickness(diameter, htd)
        
        return diameter - 2*weff

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef effective_esl_thickness(self, double diameter, double htd):
        """Returns both the effective width of the endothelial surface layer 
        (which depends on the discharge hematocrit), according to the empirical
        model of Pries and Secomb in their 2005 paper 'Microvascular blood 
        viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
               htd: Discharge hematocrit.
        OUTPUT: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double sf, doff, dcrit, d50, eamp, ewidth, epeak, ehd, wmax

        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        diameter = diameter / sf
        
        # Parameters optimized for a hematocrit dependent ESL-impact on flow:
        doff = 2.4
        dcrit = 10.5
        d50 = 100.
        eamp = 1.1
        ewidth = 0.03
        epeak = 0.6
        ehd = 1.18
        wmax = 2.6

        if diameter <= doff:
            was = 0.
            wpeak = 0.
        elif diameter <= dcrit:
            wpeak = eamp * (diameter-doff) / (dcrit-doff)
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
        else:
            wpeak = eamp * exp(-ewidth*(diameter - dcrit))
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax

        return (was + wpeak * (1 + htd * ehd)) * sf
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
