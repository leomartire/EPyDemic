from scipy.spatial import distance_matrix
import numpy as np

class Population:
    def __init__(self, X, percent_medic):
        self.N = np.size(X, 0)
        self.X = np.zeros((self.N,2),dtype=float)
        self.ages = np.zeros(self.N,dtype=int)
        self.role = np.zeros(self.N,dtype=int)
        self.status = np.zeros(self.N,dtype=int)
        # 0 = healthy
        # 1 = exposed
        # 2 = symptomatic
        # 3 = hospitalised
        # 4 = immune
        # 5 = dead
        self.since = np.zeros(self.N,dtype=int)
        self.workplace = np.ones(self.N,dtype=int)*(-1)
        self.hospital = np.ones(self.N,dtype=int)*(-1)
        self.initialise(X, percent_medic)
        # counters
        self.died_alone = 0
        self.died_hospital = 0
        self.nb_friends_p0 = 0
        return(None)
    def initialise(self, X, percent_medic):
        self.set_posi(X)
        self.init_ages(percent_medic)
        self.set_role()
        self.set_heal('all')
        self.init_relations()
    def __str__(self):
        tit = ' ID                  position age role status since workplace  hospital\n'
        vals = ('\n'.join([('%3d' % (i))+' '+
                           str(self.X[i,:])+' '+
                           ('%3d' % (self.ages[i]))+' '+
                           ('%4d' % (self.role[i]))+' '+
                           ('%6d' % (self.status[i]))+' '+
                           ('%5d' % (self.since[i]))+' '+
                           ('%9d' % (self.workplace[i]))+' '+
                           ('%9d' % (self.hospital[i]))
                           for i in range(self.N)]))
        return(tit + vals)
    def set_posi(self, XP):
        self.X = XP
        return(0)
    def init_ages(self, percent_medic):
        from globals import verbose
        from utility import age_curve_sample_many
        self.ages = age_curve_sample_many(self.N)
        #re-sample until valid medics ages are found
        self.NMedics = int(np.ceil(self.N*percent_medic/100))
        n_valid_medics = np.size(self.get_ids_valid_medics())
        if(verbose):
            print(str(n_valid_medics)+' valid medics, need '+str(self.NMedics))
        while(not(n_valid_medics>=self.NMedics)):
            n_valid_medics = np.size(self.get_ids_valid_medics())
            stop
        return(0)
    # Roles.
    def set_role(self):
        self.role = np.zeros(self.N, dtype=int)
        id_possible_medics = self.get_ids_valid_medics()
        id_medics = id_possible_medics[0][0:self.NMedics]
        self.role[id_medics] = 1
        return(0)
    def get_ids_valid_medics(self):
        return(np.where(np.logical_and(self.ages>=25, self.ages<=65)))
    def get_medics(self):
        # return available medics
        avail_medics = np.logical_and(self.role==1,
                                      np.logical_and(np.logical_or(self.status==0,
                                                                   self.status==4),
                                                     self.workplace==-1))
        
        return(np.ndarray.flatten(np.argwhere(avail_medics)))
    def set_workplace(self, subset, hospitalID):
        self.workplace[subset] = hospitalID
    # Friends.
    def init_relations(self):
        from globals import F_maxDist, F_chanceInit
        from utility import remove_extra_friends
        D = distance_matrix(self.X, self.X)
        mask_distance = np.logical_and(D < F_maxDist, D>0.)
        D[mask_distance] = (np.random.rand(np.size(D[mask_distance]))*100. < F_chanceInit)
        D[np.logical_not(mask_distance)] = 0.
        D = D + D.T - np.diag(D.diagonal()) # symmetrize     
        D = np.array(np.where(D!=0, D/D, 0), dtype=bool) # normalise (where symmetrisation lead to having 2's)
        D = remove_extra_friends(D) # remove extra friends in one direction
        D = remove_extra_friends(D.T).T # remove extra friends in the other direction
        D = np.array(D, dtype=bool)
        self.relationships = D
        return(0)
    def get_friends_list(self):
        return(np.argwhere(np.tril(self.relationships)))
    # Status changes.
    def update_since(self):
        self.since += 1
        return(0)
    def set_status(self, subset, value):
        if(isinstance(subset,str) and subset=='all'):
            self.status = np.ones(self.N,dtype=int) * value
        else:
            self.status[subset] = value
        return(0)
    def reset_since(self, subset):
        if(isinstance(subset,str) and subset=='all'):
            self.since = np.zeros(self.N,dtype=int)
        else:
            self.since[subset] = 0
        return(0)
    def set_all_healthy(self):
        self.set_status(self, 'all', 0)
        self.reset_since('all')
        return(0)
    def set_heal(self, subset): self.set_status(subset, 0); self.reset_since(subset); return(0)
    def set_expo(self, subset): self.set_status(subset, 1); self.reset_since(subset); return(0)
    def set_symp(self, subset): self.set_status(subset, 2); self.reset_since(subset); return(0)
    def set_hosp(self, subset): self.set_status(subset, 3); self.reset_since(subset); return(0)
    def set_immu(self, subset): self.set_status(subset, 4); self.reset_since(subset); return(0)
    def set_dead(self, subset): self.set_status(subset, 5); self.reset_since(subset); return(0)
    def get_heal(self): return(np.ndarray.flatten(np.argwhere(self.status==0)))
    def get_expo(self): return(np.ndarray.flatten(np.argwhere(self.status==1)))
    def get_symp(self): return(np.ndarray.flatten(np.argwhere(self.status==2)))
    def get_hosp(self): return(np.ndarray.flatten(np.argwhere(self.status==3)))
    def get_immu(self): return(np.ndarray.flatten(np.argwhere(self.status==4)))
    def get_dead(self): return(np.ndarray.flatten(np.argwhere(self.status==5)))
    def patient_0(self):
        p0 = np.random.randint(0, self.N)
        self.status[p0] = 1
        friends = self.get_friends_list()
        self.nb_friends_p0 = np.size(np.argwhere(friends==p0),0)
        return(0)
    def contaminate(self):
        from globals import F_chanceCont
        friends = self.get_friends_list()
        vulnerable = (self.status[friends]==0)
        vulnerable = np.logical_or(vulnerable[:,0],vulnerable[:,1])
        dangerous = np.logical_or(self.status[friends]==1,self.status[friends]==2)
        dangerous = np.logical_or(dangerous[:,0],dangerous[:,1])
        action = np.logical_and(vulnerable, dangerous)
        contaminable = np.unique(np.ndarray.flatten(friends[action])) # grab every vulnerable couple
        contaminable = contaminable[self.status[contaminable]==0] # grab only healthy ones
        contaminable = contaminable[np.random.rand(np.size(contaminable))*100. < F_chanceCont]
        self.set_expo(contaminable)
        return(0)
    def expo2symp(self, HCR):
        from globals import T_timeBeforeSymp, verbose
        exposed = self.get_expo()
        getting_sick = exposed[self.since[exposed] > T_timeBeforeSymp]
        self.set_symp(getting_sick)
        medics_getting_sick = np.ndarray.flatten(np.argwhere(np.logical_and(self.role[getting_sick]==1,
                                                                            self.workplace[getting_sick]!=-1)))
        if(verbose):
            if(np.size(medics_getting_sick)>0):
                print('medics getting sick: '+str(exposed[medics_getting_sick]))
        self.disaffect_medics_getting_sick(exposed[medics_getting_sick], HCR)
        return(0)
    def symp2fate(self):
        from globals import T_DaS, T_DaS_sig, C_tippingAge, C_survAloneBase, C_survAloneLowe
        from utility import chance_age
        symptomatic = self.get_symp()
        # Choose only the ones at end of sick period.
        symptomatic = symptomatic[np.ndarray.flatten(np.argwhere(self.since[symptomatic]>=T_DaS + (np.random.rand()-0.5)*T_DaS_sig))]
        chance = chance_age(self.ages[symptomatic], C_tippingAge, C_survAloneBase, C_survAloneLowe)
        isAlive = (np.random.rand(np.size(symptomatic))*100. <= chance)
        self.set_immu(symptomatic[np.ndarray.flatten(np.argwhere(isAlive))])
        self.set_dead(symptomatic[np.ndarray.flatten(np.argwhere(np.logical_not(isAlive)))])
        if(np.size(isAlive)-np.sum(isAlive) > 0):
            self.died_alone += np.size(isAlive)-np.sum(isAlive)
        return(0)
    def hosp2fate(self, HCR):
        from globals import T_DiH, T_DiH_sig, C_tippingAge, C_survHospBase, C_survHospLowe
        from utility import chance_age
        hospitalised = self.get_hosp()
        # Choose only the ones at end of sick period.
        hospitalised = hospitalised[np.ndarray.flatten(np.argwhere(self.since[hospitalised]>=T_DiH + (np.random.rand()-0.5)*T_DiH_sig))]
        if(np.size(hospitalised)>0):
            chance = chance_age(self.ages[hospitalised], C_tippingAge, C_survHospBase, C_survHospLowe)
            isAlive = (np.random.rand(np.size(hospitalised))*100. <= chance)
            self.set_immu(hospitalised[np.ndarray.flatten(np.argwhere(isAlive))])
            self.set_dead(hospitalised[np.ndarray.flatten(np.argwhere(np.logical_not(isAlive)))])
            if(np.size(isAlive)-np.sum(isAlive) > 0):
                self.died_hospital += np.size(isAlive)-np.sum(isAlive)
            HCR.remove_patients(hospitalised) # remove everyone in any case
        return(0)
    def disaffect_medics_getting_sick(self, subset, HCR):
        self.set_workplace(subset, -1) # population side
        for p in subset:
            HCR.Carers[HCR.Carers==p]=-1 # hospital side
        return(0)
    # Plot map.
    def plotMap(self, day=0):
        import matplotlib.pyplot as plt
        from plots import choice_colours
        friends_linewidth = np.max([8./self.N, 0.08])
        person_size = np.max([2000./self.N, 40])
        mark = '.'; mark_dead = 'x'
        friends = self.get_friends_list()
        for f in friends:
            plt.plot([self.X[f[0], 0], self.X[f[1], 0]], [self.X[f[0], 1], self.X[f[1], 1]], 'k:', linewidth=friends_linewidth)
        (col_heal, col_expo, col_symp, col_hosp, col_immu, col_dead) = choice_colours()
        plt.scatter(self.X[self.get_heal(),0], self.X[self.get_heal(),1], person_size, color=col_heal, marker=mark)
        plt.scatter(self.X[self.get_expo(),0], self.X[self.get_expo(),1], person_size, color=col_expo, marker=mark)
        plt.scatter(self.X[self.get_symp(),0], self.X[self.get_symp(),1], person_size, color=col_symp, marker=mark)
        plt.scatter(self.X[self.get_hosp(),0], self.X[self.get_hosp(),1], person_size, color=col_hosp, marker=mark)
        plt.scatter(self.X[self.get_immu(),0], self.X[self.get_immu(),1], person_size, color=col_immu, marker=mark)
        plt.scatter(self.X[self.get_dead(),0], self.X[self.get_dead(),1], person_size, color=col_dead, marker=mark_dead)
        plt.title('Day '+str(day))
        plt.xlabel('longitude [°]')
        plt.ylabel('latitude [°]')
        #plt.show()
    # Count.
    def count_status(self):
        return(np.array([np.sum(self.status==i) for i in range(6)], dtype=int))

class Healthcare:
    def __init__(self, NHOS, POP):
        from globals import HOSPITALS_TOULOUSE, H_bedNumber, H_bedPerCarer
        self.N = NHOS
        XH = HOSPITALS_TOULOUSE[np.arange(self.N),:]
        self.X = XH
        self.NBeds = np.ones(self.N, dtype=int) * H_bedNumber
        self.NBedsPerCarer = np.ones(self.N, dtype=int) * H_bedPerCarer
        self.NeededCarers = np.array(np.ceil(self.NBeds/self.NBedsPerCarer), dtype=int)
        self.Carers = np.ones((self.N, np.max(self.NeededCarers)), dtype=int) * (-1)
        self.assign_workers(POP)
        self.Patients = np.ones((self.N, np.max(self.NBeds)), dtype=int) * (-1)
    def assign_workers(self, POP: Population):
        from globals import verbose
        availMedics = POP.get_medics()
        availMedicsIsFree = np.array(np.ones(np.size(availMedics)),dtype=bool)
        #print(availMedics)
        #print(availMedicsIsFree)
        NCarersMissing = self.NeededCarers - self.get_nb_carers()
        #print(NCarersMissing)
        NMedicsPerHospitalTry = np.array(np.ceil(np.size(availMedics)*NCarersMissing/np.sum(NCarersMissing)), dtype=int)
        #print(NMedicsPerHospitalTry)
        for hi in range(self.N):
            #print(hi)
            curIDs = np.ndarray.flatten(np.argwhere(availMedicsIsFree)) # Find free medics.
            curIDs = curIDs[np.arange(np.min([NMedicsPerHospitalTry[hi], np.size(curIDs)]))] # Select the N first needed for this hospital.
            #print(curIDs)
            if(np.size(curIDs)>0):
                medicsForThisHospital = availMedics[curIDs] # Get the actual IDs.
                #print(medicsForThisHospital)
                IDsFreeCarerSlots = np.ndarray.flatten(np.argwhere(self.Carers[hi,:]==-1))
                #print(IDsFreeCarerSlots)
                if(np.size(IDsFreeCarerSlots)==1):
                    IDsToFill = IDsFreeCarerSlots
                else:
                    IDsToFill = IDsFreeCarerSlots[0:np.min([np.size(IDsFreeCarerSlots), np.size(medicsForThisHospital)])]
                #print(IDsToFill)
                #print(medicsForThisHospital)
                medicsForThisHospital = medicsForThisHospital[np.arange(np.size(IDsToFill))]
                #print(medicsForThisHospital)
                #print(self.Carers)
                self.Carers[hi, IDsToFill] = medicsForThisHospital # Assign carers to hospital.
                POP.set_workplace(medicsForThisHospital, hi)
                #print(self.Carers)
                availMedicsIsFree[curIDs] = False # Set them as not free anymore.
            else:
                if(verbose):
                    print('no more medics available for hospital '+str(hi))
    def get_carers(self, hID):
        # Get carers in hospital hID.
        return(np.ndarray.flatten(self.Carers[hID,np.where(self.Carers[hID,:]!=-1)]))
    def get_nb_carers(self):
        return(np.array([np.size(self.get_carers(i)) for i in range(self.N)],dtype=int))
    def get_patients(self, hID):
        # Get carers in hospital hID.
        return(np.ndarray.flatten(self.Patients[hID,np.where(self.Patients[hID,:]!=-1)]))
    def get_nb_patients(self):
        return(np.array([np.size(self.get_patients(i)) for i in range(self.N)],dtype=int))
    def get_free_spaces(self):
        return(self.get_nb_carers() * self.NBedsPerCarer - self.get_nb_patients())
    def try_hospitalise(self, POP):
        from globals import H_maxDist, verbose
        symptomatics = POP.get_symp()
        D = distance_matrix(POP.X[symptomatics, :], self.X)
        D = (D<=H_maxDist)
        #print(D)
        for si in range(np.size(symptomatics,0)):
            #print(D[si,:])
            freeSpaces = self.get_free_spaces()
            #print(freeSpaces)
            possible_hospitals = np.ndarray.flatten(np.argwhere(np.array(np.logical_and(D[si,:],freeSpaces>0))))
            if(np.size(possible_hospitals)>0):
                chosen_hospital = possible_hospitals[0]
                patientSlots = np.ndarray.flatten(np.argwhere(self.Patients[chosen_hospital]==-1))
                if(np.size(patientSlots)==0):
                    if(verbose):
                        print('no bed or carer for this patient'+str(chosen_hospital))
                else:
                    self.Patients[chosen_hospital, patientSlots[0]] = symptomatics[si]
                    POP.set_hosp(symptomatics[si])
                    POP.hospital[symptomatics[si]] = chosen_hospital
            else:
                if(verbose):
                    print('no hospital can welcome patient '+str(symptomatics[si])+'.')
        return(0)
    def remove_patients(self, subset):
        IDs = np.array([np.ndarray.flatten(np.argwhere(self.Patients==subset[i])) for i in range(np.size(subset))])
        #print(self.Patients)
        #print(subset)
        #print(IDs)
        for ij in IDs:
            self.Patients[ij[0], ij[1]] = -1
        return(0)
    # Count.
    def get_statistics(self):
        usable = np.sum(self.get_nb_carers()*self.NBedsPerCarer)/np.sum(self.NBeds)
        used = np.sum(self.get_nb_patients())/np.sum(self.NBeds)
        return(np.array([usable, used]))