import numpy as np
import matplotlib.pyplot as plt

def choice_colours():
    col_heal = np.array([0.3,0.8,0.9])
    col_expo = np.array([0.7,0.3,0.3])
    col_symp = np.array([0.7,0.1,0.3])
    col_hosp = np.array([1,0.6,0.6])
    col_immu = np.array([0.3, 0.8, 0.3])
    col_dead = np.array([1,1,1])*0.2
    return(col_heal, col_expo, col_symp, col_hosp, col_immu, col_dead)

def plot_area_results_wrapper(pop_counters_in):
    pop_counters = np.copy(pop_counters_in)
    NPOP = np.unique(np.sum(pop_counters[0,0,:]))
    if(np.size(NPOP)>1):
        print(NPOP)
        stop
    else:
        NDAYSTORUN = np.size(pop_counters,1)
        pop_counters = np.mean(pop_counters, 0) *100 / NPOP # make percent
        days_x=np.linspace(0, NDAYSTORUN, NDAYSTORUN, dtype=int)
        if(np.any(np.sum(pop_counters,1)==0)):
            zerosumid = np.ndarray.flatten(np.argwhere(np.sum(pop_counters,1)==0))
            #print(zerosumid[0])
            days_x[zerosumid[0]:] = days_x[zerosumid[0]-1]
            days_x[-1] = NDAYSTORUN
        plot_area_results(days_x,
                          pop_counters[:,0], pop_counters[:,1]+pop_counters[:,2],
                          pop_counters[:,3], pop_counters[:,4], pop_counters[:,5],
                          True,NPOP)

def plot_area_results(dayz, healthy, infected, hospitalized, immunised, dead, make_percents,NPOP):
    (col_heal, col_expo, col_symp, col_hosp, col_immu, col_dead) = choice_colours()
    # area plot
    l1 = dead; labl1 = 'dead'; coll1 = col_dead
    l2 = l1 + infected; labl2 = 'infected'; coll2 = col_symp
    l3 = l2 + hospitalized; labl3 = 'hospitalized'; coll3 = col_hosp
    l4 = l3 + immunised; labl4 = 'immunised'; coll4 = col_immu
    l5 = l4 + healthy; labl5 = 'healthy'; coll5 = col_heal
    plt.fill_between(dayz, l5, label=labl5, color=coll5)
    plt.fill_between(dayz, l4, label=labl4, color=coll4)
    plt.fill_between(dayz, l3, label=labl3, color=coll3)
    plt.fill_between(dayz, l2, label=labl2, color=coll2)
    plt.fill_between(dayz, l1, label=labl1, color=coll1)
    plt.legend()
    if(make_percents):
        plt.ylim([0, 100])
    else:
        plt.ylim([0, NPOP])
    plt.xlim([np.min(dayz), np.max(dayz)])
    plt.xlabel('days')
    plt.ylabel('%'+(' of total population (%d)'%(NPOP)))
    #plt.show()