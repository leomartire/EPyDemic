import numpy as np
from utility import *

def km2deg(km):
    return(km*1/111)

# global vars
verbose = 0
#IDPERSON = 1 # Only used in version 1.
#IDHOSPITAL = 1 # Only used in version 1.

# Relationships.
F_maxNumber      = 20 # Maximum number of friends one can have.
F_maxDist        = 40. *1./111. # Maximum distance [°] friends can be separated by (1° = 111km).
F_chanceInit     = 10. # Chance [%] two people can become friends.
F_chanceCont     = 65. # Chance [%] one friend has to contaminate the other.

# Hospital parameters.
H_maxDist        = 100 *1./111. # Maximum distance within which a symptomatic individual can be hospitalised.
H_bedNumber      = 100 # Number of beds (possible patients) per hospital.
H_bedPerCarer    = 10 # Number of beds one medic can take care of.

# Chances.
C_survAloneBase  = 80 # Chance [%] of surviving when alone, base (at youngest).
C_survAloneLowe  = 66 # Chance [%] of surviving when alone, lowest (at oldest).
C_survHospBase   = 95 # Chance [%] of surviving when hospitalised, base (at youngest).
C_survHospLowe   = 90 # Chance [%] of surviving when hospitalised, lowest (at oldest).
C_tippingAge     = 60 # Age at which chances start to tip towards the lowest value.

# Timings.
#T_minAgeSymp     = 0 # Age after which someone can transition from exposed to symptomatic. Unused in version 2.
T_timeBeforeSymp = 14 # Number of days spent contaminated without symptoms.
T_DiH            = 10 # Numbers of days to spend in hospital before deciding fate (i.e. become dead or immune).
T_DiH_sig        = 3 # Random variation amplitude for T_DiH.
T_DaS            = 7 # Numbers of days to spend without treatment before deciding fate (i.e. become dead or immune).
T_DaS_sig        = 5 # Random variation amplitude for T_DaS.

CITIES_TOULOUSE = np.array([[43.6006786,1.3628016], # Toulouse
                            [44.0216736,1.2945156], # Montauban
                            [43.9289852,2.097281] # Albi
                           ])
CITIES_TOULOUSE_SIG = np.array([km2deg(6),
                                km2deg(3),
                                km2deg(3.5)
                               ]) / .5

HOSPITALS_TOULOUSE = np.fliplr(np.array([[43.5591666,1.4513873], # Toulouse Rangueil
                                         [44.0228025,1.3495677], # Montauban 1
                                         [43.9256706,2.1406693], # Albi 1
                                         [43.6080627,1.3956933], # Toulouse Purpan
                                         [44.0063895,1.3771156], # Montauban 2
                                         [43.9375238,2.1661137], # Albi 2
                                         [43.6010239,1.4314358], # Toulouse La Grave
                                        ]))