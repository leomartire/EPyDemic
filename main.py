import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utility import *
from classes import *
from globals import *
import time
from plots import *

matplotlib.rc('axes',edgecolor='k',facecolor='w',linewidth=2,xmargin=0,ymargin=0)
#matplotlib.rc('axes',autolimit_mode='round_numbers')
matplotlib.rc('xtick.major',width=2,size=8)
matplotlib.rc('ytick.major',width=2,size=8)
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
matplotlib.rc('legend',fontsize=16, handlelength=2)
#matplotlib.rc('text', usetex=False)
matplotlib.rc('text', usetex=False)

def iterate_new(NDAYSTORUN, NRUNS=1, doPlotMap=False):
    from globals import F_chanceCont
    DA_DH_FP0 = np.zeros((NRUNS,3))
    pop_counters = np.zeros((NRUNS, NDAYSTORUN, 6))
    hcr_counters = np.zeros((NRUNS, NDAYSTORUN, 2))
    DA_DH = np.zeros((NRUNS, NDAYSTORUN, 2))
    for run in range(NRUNS):
        print('Run n°'+str(run)+'.')
        POP = Population(XP, percent_medics)
        HCR = Healthcare(NHOS, POP)
        HCR.assign_workers(POP)
        # POP.plotMap()
        
        for day in range(NDAYSTORUN):
            if(day==1):
                POP.patient_0()
            print('> Day ' +str(day)+'.')
            #if(day>5):
            #    F_chanceCont = 0
            POP.update_since()
            # contaminate
            POP.contaminate()
            # upgrade expo to symp (and remove symptomatic carers from hospitals)
            POP.expo2symp(HCR)
            # update hospital population (heal, or kill)
            POP.hosp2fate(HCR)
            # try re-assigning healthy medics (if one was healed, or if there were too much at start and now some are dead)
            HCR.assign_workers(POP)
            # try to send symptomatics to hospital
            HCR.try_hospitalise(POP)
            # update symptomatic population
            POP.symp2fate()
            # print('day '+str(day))
            # print(str(POP))
            pop_counters[run, day, :] = POP.count_status()
            hcr_counters[run, day, :] = HCR.get_statistics()
            DA_DH[run, day, :] = [POP.died_alone, POP.died_hospital]
            if(doPlotMap):
                fig = plt.figure(figsize=(16,10))
                plt.subplot(2,2,1)
                POP.plotMap(day)
                plt.subplot(2,2,2)
                plot_area_results_wrapper(pop_counters)
                plt.subplot(2,2,3)
                
                (__, __, __, col_hosp, __, __) = choice_colours()
                hcr_counters_p = np.zeros((day+1, 2))
                hcr_counters_p = hcr_counters[run, 0:(day+1), :]*100
                days_x=np.arange(0,day+1)
                plt.fill_between(days_x, hcr_counters_p[:, 1], color=col_hosp,label='total occupation')
                plt.plot(days_x, hcr_counters_p[:, 0], linewidth=3, color='k',label='healing capacity')
                plt.xlim([0,NDAYSTORUN-1])
                plt.ylim([0,100])
                plt.ylabel('%'+(' of total capacity (%d beds)'%(np.sum(HCR.NBeds))))
                plt.xlabel('days')
                plt.legend()
                
                plt.subplot(2,2,4)
                
                (__, __, __, __, __, col_dead) = choice_colours()
                days_x=np.arange(0,day+1)
                percent_death_alone = 100*DA_DH[run,0:(day+1),0]/pop_counters[run,0:(day+1),-1]
                percent_death_hospi = 100*DA_DH[run,0:(day+1),1]/pop_counters[run,0:(day+1),-1]
                plt.plot(days_x, percent_death_alone, color=col_dead+[0.3,0,0], linewidth=3, label='died alone')
                plt.plot(days_x, percent_death_hospi, color=col_dead, linewidth=3, label='died in hospital')
                plt.xlim([0,NDAYSTORUN-1])
                plt.xlabel('days')
                plt.ylim([0,100])
                plt.ylabel('%' + (' of total deaths (%d)' % (pop_counters[run,day,-1])))
                plt.legend()
                
                plt.savefig(('plots/n=%05d_day=%05d.png' % (POP.N, day)), dpi=300, quality=95, bbox_inches='tight')
                plt.show()
                plt.close()
        DA_DH_FP0[run, 0] = POP.died_alone
        DA_DH_FP0[run, 1] = POP.died_hospital
        DA_DH_FP0[run, 2] = POP.nb_friends_p0
    plt.ion()
    return(POP, HCR, pop_counters, hcr_counters, DA_DH_FP0)

# Simulation parameters.
NDAYSTORUN = 61
NPOP = 500
NHOS = 3
NRUNS = 1; doPlotMap = True; plt.ioff();
#NRUNS = 1; doPlotMap = True; plt.ion(); # for debug
#NRUNS = 1; doPlotMap = False; plt.ion() # for mass estimations

# Compute percent medics
#6706400 from https://en.wikipedia.org/wiki/Demographics_of_France
# 226000 from https://drees.solidarites-sante.gouv.fr/IMG/pdf/dossier_presse_demographie.pdf
percent_medics = 1* 100 * 226000/6706400
#percent_medics = 10

# Initialise positions.
start = time.time()
XP = initialise_positions(NPOP)
end = time.time(); print( 'initialise_positions       '+str(end-start) );

(POP, HCR, pop_counters, hcr_counters, RES) = iterate_new(NDAYSTORUN, NRUNS=NRUNS, doPlotMap=doPlotMap)
#POP.plotMap()
RES = np.mean(RES, 0)
#plot_area_results_wrapper(pop_counters)
ndeaths = np.mean(pop_counters[:,-1,5], 0)
print('%4d (%3.0f percent of deaths) died wihout hospital care. %4d (%3.0f percent of deaths) died in hospital care.'
      % (RES[0], 100*RES[0]/ndeaths, RES[1], RES[1]*100/ndeaths))
print('Patient 0 had %4d friends on average.'
      % (RES[2]))