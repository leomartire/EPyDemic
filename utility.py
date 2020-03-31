import numpy as np
from scipy.special import erf
import random
#from classes import *
from scipy.integrate import simps
import os
from globals import *
from scipy.interpolate import interp2d, RegularGridInterpolator

def sample_from_histogram(bins_x, bins_y):
    datacumsum = np.cumsum(bins_y) # "store" bins end to end
    choice = np.random.randint(1,datacumsum[-1]+1) # generate random number from very start to total samples
    thebin = np.argwhere(datacumsum>=choice) # find bins including or containing random choice
    return(bins_x[thebin[0]-1]) # get x value of first bin to verify condition

def sample_from_histogram_many(bins_x, bins_y, N):
    samples = np.zeros(N)
    for c in range(N):
        samples[c] = sample_from_histogram(bins_x, bins_y)
    return(samples)

def remove_extra_friends(D_in):
    from globals import F_maxNumber
    D = np.copy(D_in)
    tooManyFriends = np.ndarray.flatten(
                       np.argwhere(
                         np.array([np.sum(D[i,:])>F_maxNumber
                                   for i in range(np.size(D,0))], dtype=bool)))
    for tmf in tooManyFriends:
        #print('D['+str(tmf)+',:] '+str(D[tmf,:]))
        #friendsIDs = np.ndarray.flatten(np.argwhere(D[tmf,:]))
        #print('fids '+str(friendsIDs))
        #IDsToSetToZero = np.arange(friendsIDs[F_maxNumber],np.size(D,1))
        #print('idst0 '+str(IDsToSetToZero))
        #D[tmf,IDsToSetToZero]=False # arbitrarily unfriend the last ones
        D[tmf,np.arange(np.ndarray.flatten(np.argwhere(D[tmf,:]))[F_maxNumber],np.size(D,1))]=False # arbitrarily unfriend the last ones
    return(D)

def samples_from_2dpdf(xv, yv, density, N):
    # 2D sampling from a PDF using the rejection method (https://fr.wikipedia.org/wiki/M%C3%A9thode_de_rejet).
    x = xv[1,:]; y = yv[:,1]
    density = density/simps(simps(density, x), y) # normalise pdf
    maxd = np.max(density)
    interpolant = interp2d(x, y, density, kind='linear') # 0.1374976348876953
    #interpolant = interp2d(x, y, density, kind='cubic') # 0.14160812377929688
    #interpolant = interp2d(x, y, density, kind='quintic') # 0.14267345428466796
    samples = np.zeros((N,2))
    for i_sample in range(N):
        found = False
        while(not(found)):
            Y_x = np.random.uniform(low = np.min(xv), high = np.max(xv), size=(1))
            Y_y = np.random.uniform(low = np.min(yv), high = np.max(yv), size=(1))
            f_of_Y = interpolant(Y_x, Y_y)
            if(np.random.rand(1)*maxd > f_of_Y):
                found = False
            else:
                found = True
        samples[i_sample, :] = [Y_x, Y_y]
    return(samples)

def age_curve_sample():
    # https://www.insee.fr/en/statistiques/3312960
    age_pyramid_0_to_100 = np.array([706382, 716159, 729139, 749142, 770897, 795049, 801336, 818973, 824266, 844412, 836610, 841774, 833484, 847250, 828874, 828224, 825535, 824243, 830859, 832135, 778595, 767419, 738255, 741493, 731720, 709814, 710229, 747365, 762740, 783278, 793756, 805709, 809462, 824388, 823154, 817616, 809113, 860183, 868514, 876362, 830619, 812560, 815529, 795012, 818506, 859407, 905508, 925828, 921091, 900389, 888940, 878137, 872944, 891913, 893796, 901416, 889289, 857860, 858184, 852627, 845836, 827046, 818270, 809103, 799407, 795066, 776073, 784280, 760998, 783527, 766434, 759622, 739203, 692884, 518955, 502516, 483835, 443448, 389310, 397453, 408011, 390052, 372609, 362050, 336284, 325338, 293641, 280250, 250255, 226053, 186015, 160562, 132403, 110466, 89330, 69801, 53201, 39728, 29030, 20035, 21860])
    ages = np.linspace(0,100,100,dtype=int)
    return(sample_from_histogram(ages, age_pyramid_0_to_100))

def age_curve_sample_many(N):
    # https://www.insee.fr/en/statistiques/3312960
    age_pyramid_0_to_100 = np.array([706382, 716159, 729139, 749142, 770897, 795049, 801336, 818973, 824266, 844412, 836610, 841774, 833484, 847250, 828874, 828224, 825535, 824243, 830859, 832135, 778595, 767419, 738255, 741493, 731720, 709814, 710229, 747365, 762740, 783278, 793756, 805709, 809462, 824388, 823154, 817616, 809113, 860183, 868514, 876362, 830619, 812560, 815529, 795012, 818506, 859407, 905508, 925828, 921091, 900389, 888940, 878137, 872944, 891913, 893796, 901416, 889289, 857860, 858184, 852627, 845836, 827046, 818270, 809103, 799407, 795066, 776073, 784280, 760998, 783527, 766434, 759622, 739203, 692884, 518955, 502516, 483835, 443448, 389310, 397453, 408011, 390052, 372609, 362050, 336284, 325338, 293641, 280250, 250255, 226053, 186015, 160562, 132403, 110466, 89330, 69801, 53201, 39728, 29030, 20035, 21860])
    ages = np.linspace(0,100,100,dtype=int)
    return(sample_from_histogram_many(ages, age_pyramid_0_to_100,N))

def chance_age(age, C_tippingAge, CHANCE_BASE, CHANCE_LOWEST):
    return( 0.5 * (1-erf((age-C_tippingAge)/10)) * (CHANCE_BASE-CHANCE_LOWEST) + CHANCE_LOWEST )

def initialise_positions(N):
    file_population = ('POSITIONS_%07d' % (N))
    if(os.path.exists(file_population+'.npy')):
        XP = np.load(file_population+'.npy')
    else:
        latmin = 43.326075 ; latmax = 44.072278
        lonmin = 0.878051; lonmax = 2.376380
        cities = CITIES_TOULOUSE
        cities_sig = CITIES_TOULOUSE_SIG
        cities = np.fliplr(cities)
        dx = 0.001; x = np.arange(lonmin,lonmax+dx,dx); y = np.arange(latmin,latmax+dx,dx);
        (xv, yv) = np.meshgrid(x, y)
        density = 0*xv + 0.02
        for i in range(0,np.size(cities,0)):
            density += np.exp( -(np.power(xv-cities[i,0],2)+np.power(yv-cities[i,1],2))/np.power(cities_sig[i],2) );
        XP = samples_from_2dpdf(xv, yv, density, N)
        np.save(file_population, XP)
        #plt.pcolor(xv,yv,density); plt.colorbar(); plt.show()
        #plt.hist2d(XP[:,0], XP[:,1], bins=50); plt.show()
    return(XP)

def pick_random_name():
    firstnamelist = ['Adrien', 'Alain', 'Albert', 'Alexandre', 'Alexis', 'Alice', 'Anaïs', 'André', 'Andrée', 'Anne', 'Annick', 'Annie', 'Anthony', 'Antoine', 'Arnaud', 'Arthur', 'Audrey', 'Aurélie', 'Benjamin', 'Benoît', 'Bernadette', 'Bernard', 'Brigitte', 'Bruno', 'Béatrice', 'Camille', 'Caroline', 'Catherine', 'Cedric', 'Chantal', 'Charles', 'Charlotte', 'Chloé', 'Christelle', 'Christian', 'Christiane', 'Christine', 'Christophe', 'Claire', 'Claude', 'Claudine', 'Clément', 'Colette', 'Corinne', 'Cécile', 'Céline', 'Damien', 'Daniel', 'Danielle', 'David', 'Delphine', 'Denis', 'Denise', 'Didier', 'Dominique', 'Dominique', 'Eliane', 'Elisabeth', 'Elise', 'Elodie', 'Emile', 'Emilie', 'Emma', 'Emmanuel', 'Enzo', 'Eric', 'Evelyne', 'Fabien', 'Fabrice', 'Fernand', 'Florence', 'Florian', 'Francis', 'Franck', 'François', 'Françoise', 'Frédéric', 'Gabriel', 'Geneviève', 'Georges', 'Georgette', 'Germaine', 'Gilbert', 'Gilles', 'Ginette', 'Gisèle', 'Guillaume', 'Guy', 'Gérard', 'Henri', 'Henriette', 'Hervé', 'Hugo', 'Hélène', 'Isabelle', 'Jacqueline', 'Jacques', 'Jean', 'Jean-Claude', 'Jean-Luc', 'Jean-Pierre', 'Jeanne', 'Jeannine', 'Jeremy', 'Jerome', 'Joseph', 'Josette', 'Josiane', 'Joël', 'Jules', 'Julie', 'Julien', 'Juliette', 'Karine', 'Kevin', 'Laetitia', 'Laura', 'Laurence', 'Laurent', 'Louis', 'Louise', 'Lucas', 'Lucie', 'Lucien', 'Lucienne', 'Ludovic', 'Léa', 'Léon', 'Madeleine', 'Manon', 'Marc', 'Marcel', 'Marcelle', 'Marguerite', 'Maria', 'Marie', 'Marine', 'Marion', 'Marthe', 'Martine', 'Mathieu', 'Mathilde', 'Maurice', 'Maxime', 'Michel', 'Michelle', 'Michèle', 'Mireille', 'Monique', 'Mélanie', 'Nadine', 'Nathalie', 'Nathan', 'Nicolas', 'Nicole', 'Odette', 'Olivier', 'Pascal', 'Patrice', 'Patricia', 'Patrick', 'Paul', 'Paulette', 'Pauline', 'Philippe', 'Pierre', 'Quentin', 'Raphaël', 'Raymond', 'Raymonde', 'René', 'Renée', 'Robert', 'Roger', 'Roland', 'Romain', 'Sandrine', 'Sarah', 'Serge', 'Simone', 'Sophie', 'Stéphane', 'Stéphanie', 'Suzanne', 'Sylvain', 'Sylvie', 'Sébastien', 'Thierry', 'Thomas', 'Théo', 'Thérèse', 'Valérie', 'Victor', 'Vincent', 'Virginie', 'Véronique', 'Xavier', 'Yves', 'Yvette', 'Yvonne']
    return(firstnamelist[np.random.randint(0,len(firstnamelist))])