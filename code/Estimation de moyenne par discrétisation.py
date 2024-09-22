import numpy as np
from numpy import random as rd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

'''
____________________________________________________________________________________________________________
Ce code permet de tracer les estimations, par mécanisme exponentiel, de la moyenne de n V.A. uniformes.
Le tracé est en fonction de N, nombre d'intervalles de discrétisation de l'ensemble des réponses, ou en fonction
de N et eps, degré de privacy.
Les figures de la partie 5 en proviennent.
____________________________________________________________________________________________________________
'''


'''

fonction d'utilité u simple (pour un ensemble de VA ~ U([0,1])):
'''
def u(o):
    return(-np.abs(o - 1/2))

'''
Mécanisme exponentiel de calcul de la moyenne

en entrée:  - X
            - eps
            - N le nombre d'intervalles de discrétisation
Rq: on utilise une neighboring relation de substitution (car cela donne un plus petit alpha)
'''
def MecaExpMoyenne(X, eps, N):
    n = len(X)
    #Sensitivity pour la substitution
    DeltaU = 1/n
    #ensemble des outputs
    O = [i/N for i in range(N)]
    #vecteur des probabilités (qu'on gardera a-normalisées, pour éviter les erreurs avec de petits flottants)
    probas = [np.exp(eps * u(o) / (2*DeltaU)) for o in O]
    #calcul du coefficient de normalisation
    coeffNormalis = sum(probas)

    # pour pouvoir trouver le o tiré par le sort, on ajoute à chaque élément de "probas" la 
    # somme des éléments précédents
    for i in range(1,N):
        probas[i] += probas[i-1]
    #proba a-normalisée tirée:
    probaTiree = coeffNormalis * rd.random()
    for i in range(N):
        if probaTiree <= probas[i]:
            i0 = i
            break
        if i == N-1:
            i0 = i
    return O[i0]
    
# MecaExpMoyenne(rd.random(500), 0.8, 2000)





'''
fonction traçant les résultats en fonction de N impair
en entrée:  - n nombre de V.A. voulues
            - eps
            - nbSimus tel que (5 000 000 + nbSimus) soit le double du nombre de valeurs voulues pour la courbe
            '''
def traceEstimMoyenneDeNimpair(n, eps, nbSimus):
    X = rd.random(n)
    abscisses = range(5000000, nbSimus + 1)
    #test avec N impairs
    ordonnees = [MecaExpMoyenne(X, eps, 2*N+1) for N in abscisses]
    plt.plot(abscisses, ordonnees)
    plt.show()

# traceEstimMoyenneDeNimpair(1000, 0.8, 5005000)

'''idem en fonction de N pair
'''
def traceEstimMoyenneDeNpair(n, eps, nbSimus):
    X = rd.random(n)
    abscisses = range(15000, nbSimus + 1)
    ordonnees = [MecaExpMoyenne(X, eps, 2*N) for N in abscisses]
    plt.plot(abscisses, ordonnees)
    plt.show()

# traceEstimMoyenneDeNpair(1000, 0.8, 15200)


'''même fonction mais avec une figure 3D, traçant en fonction de N et eps
en entrée:  - n nombre de V.A.
            - nbSimusN nombre d'ordonnées voulues (valeurs de N)
            - valMaxEps nombre d'abscisses voulues (valeurs de Eps)
'''

def plot_MecaExpMoyenne(n, valMaxEps, nbSimusN):
    X = rd.random(n)
    eps_values = np.arange(0.01, valMaxEps + 0.05, 0.05)
    N_values = np.arange(3, nbSimusN + 1)
    X_mesh, Y_mesh = np.meshgrid(eps_values, N_values)
    Z_mesh = np.zeros_like(X_mesh)

    for i, eps in enumerate(eps_values):
        for j, N_val in enumerate(N_values):
            Z_mesh[j, i] = MecaExpMoyenne(X, eps, N_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis')
    ax.set_xlabel('eps')
    ax.set_ylabel('N')
    ax.set_zlabel('MecaExpMoyenne')
    plt.show()

# plot_MecaExpMoyenne(1000, 3, 30)


## CONVERGENCE VERS UN MECANISME DE LAPLACE
## ___________________________________________________________________________________________________________________

'''tracé d'une distribution de Laplace
en entrée: -n
            -eps
            - nbPts le nombre de points qu'on veut entre 0 et 1
'''
def traceLaplace(n, eps, nbPts):
    #pour une relation de substitution:
    DeltaF = 1/n
    x = np.arange(0, 1, nbPts)
    laplace_values = np.random.laplace(loc= 1/2, scale=DeltaF / eps, size=nbPts)

    hist, bins, _ = plt.hist(laplace_values, bins=50, density=True, alpha=0.6, label="Distribution de Laplace")
    bin_width = bins[1] - bins[0]

    # Normalisation de l'aire des barres
    total_area = np.sum(hist) * bin_width
    plt.bar(bins[:-1], hist / total_area, width=bin_width, align='edge', alpha=0.6)


    plt.ylabel("Densité de probabilité")
    plt.legend()
    plt.grid(True)
    plt.show()

# traceLaplace(1000, 0.4, 1000)


'''
Tracé des probabilités obtenues avec notre mécanisme exponentiel d'estimation de moyenne'''
def traceExp(n,eps,N):
    #on ne prend que les abscisses compris entre 0.49 et 0.51 (pour voir en gros plan la partie intéressante)
    abscisses = [i/N for i in range(math.floor(0.49 * N), math.ceil(0.51 * N))]
    print(abscisses)
    #on utilise une sensitivity liée à une neighboring relation de substitution
    ordonnees = [np.exp(n*eps*u(absc)/2) for absc in abscisses]
    #calcul du coefficient de normalisation
    coeffNormalis = sum([np.exp(n*eps*u(i/N)/2) for i in range(N)])
    ordonnees =[y/coeffNormalis for y in ordonnees]
    plt.plot(abscisses, ordonnees)
    plt.show()

# traceExp(1000, 0.8, 10000)

'''
Tracé sur un même graphe de la distribution du mécanisme exponentiel et de la distribution de Laplace
'''
def traceSimultane(n, eps, N, nbPts):
        #pour une relation de substitution:
    DeltaF = 1/n

    #tracé de la distribution de Laplace:
    x = np.arange(0, 1, nbPts)
    laplace_values = np.random.laplace(loc= 1/2, scale=DeltaF / eps, size=nbPts)
    plt.hist(laplace_values, bins=50, density=True, alpha=0.6)

    #tracé des probabilités du méca exp:
    abscisses1 = [i/N for i in range(N)]
    #pour ne prendre que les abscisses compris entre 0.48 et 0.52 (pour voir en gros plan la partie intéressante)
    abscisses = abscisses1.copy()
    index_a_pop = 0
    for i in range(N):
        absc = abscisses1[i]
        if (absc < 0.48) or (absc > 0.52):
            abscisses.pop(index_a_pop)
        else:
            index_a_pop += 1

    #on utilise une sensitivity liée à une neighboring relation de substitution
    ordonnees = [np.exp(n*eps*u(absc)/2) for absc in abscisses]
    #calcul du coefficient de normalisation
    coeffNormalis = sum([np.exp(n*eps*u(i/N)/2) for i in range(N)])
    ordonnees =[y/coeffNormalis for y in ordonnees]
    plt.plot(abscisses, ordonnees)

    plt.grid(True)
    plt.show()


