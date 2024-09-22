import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

'''
____________________________________________________________________________________________________________
Ce code permet de calculer et de tracer des expected losses de mécanismes estimant le paramètre p de V.A. de 
Bernoulli. Certaines fonctions tracent aussi la borne inférieure donnée par la théorème de la partie 6.1, ou 
l'équivalent de l'expected loss d'un mécanisme d'inverse sensitivity donné dans la même partie.
____________________________________________________________________________________________________________
'''


'''
en entrée: -n nombre de Xi ~ B(p)
           - p le paramètre de Bernoulli
           - eps degré de privacy
           - nbEssais le nombre d'itérations pendant lesquelles on lance M sur X
en sortie: E[|M(X) - f(X)|] pour M mécanisme de Laplace de paramètre Delta_f/eps
           Rq: E[|M(X) - f(X)|] = E[|L(Delta_f/eps)|] par définition de f
           '''
def esperanceDuLossLapl(n, p, eps, nbEssais):
    X = rd.binomial(1, p, size=n)
    esperance = 0
    for essai in range(nbEssais):
        esperance += np.abs(rd.laplace(loc=0.0, scale = 1 / (n*eps), size=None))
    return(esperance/nbEssais)

# print(esperanceDuLossLapl(100, 0.5, 1, 100))


'''
Fonction traçant l'espérance calculée (plusieurs fois) par esperanceDuLoss ainsi que la droite constante 
1/ (2n(exp(2 eps)+1)) (borne inférieure donnée par le théorème 1 de l'article d'Asi & Duchi "Near 
Instance-Optimality in Differential Privacy".
Ici on prend la borne inf maximale, c'est-à-dire celle pour k=1)

en entrée:  - n, p, eps, nbEssais comme pour esperanceDuLoss
            - nbCalculs le nombre de points qu'on calcule pour tracer la courbe
'''
def traceBorneInfEsperanceLapl(n,p,eps, nbEssais, nbCalculs):
    absc = range(0, nbCalculs)
    borneInf = [1/ (2*n*(np.exp(2 * eps)+1)) for _ in absc]
    Points = [0 for _ in absc]
    for calcul in range(nbCalculs):
        Points[calcul] = esperanceDuLossLapl(n,p,eps,nbEssais)
    p1, = plt.plot(absc, Points)
    p2, = plt.plot(absc, borneInf)
    plt.legend([p1,p2], ["Expected loss", "Borne inférieure donnée par le théorème 1, avec k = 1"])
    plt.show()

# traceBorneInfEsperanceLapl(100, 0.5, 1, 100, 100)


'''
Idem qu' esperanceDuLossLapl, mais pour un mécanisme d'inverse sensitivity
'''
def esperanceDuLossInvSensi(n,p,eps,nbEssais):
    X = rd.binomial(1, p, size=n)
    #f(X)
    fX = sum(X)/n
    #vecteur des probabilités
    probas = [np.exp(-eps * np.abs(n*fX - k)/2) for k in range(0, n+1)]
    #normalisation
    probas = [proba * (1- np.exp(eps/2)) / (np.exp(-n * eps * fX/2) - np.exp(eps/2) - 1 + np.exp(n * eps * (fX - 1)/2)) for proba in probas]
    #on ajoute à chaque élément la somme des éléments précédents; ainsi, après avoir tiré un nombre aléatoire
    # dans [0, 1], on pourra successivement le comparer à chaque élément de "probas" 
    for k in range(1, n+1):
        probas[k] += probas[k-1]
    esperance = 0
    for essai in range(nbEssais):
        probaTiree = rd.random()
        for k in range(0, n):
            if probas[k] >= probaTiree:
                esperance += np.abs(k/n - fX)
                break
        if k==n-1:
            esperance += np.abs(1 - fX)    
    return(esperance/nbEssais)

# print(esperanceDuLossInvSensi(100, 0.5, 1, 100))
    


'''
Idem que traceBorneInfEsperanceLapl, mais avec le mécanisme d'inverse sensitivity
'''
def traceBorneInfEsperanceInvSensi(n,p,eps, nbEssais, nbCalculs):
    absc = range(0, nbCalculs)
    borneInf = [1/ (2*n*(np.exp(2 * eps)+1)) for _ in absc]
    Points = [0 for _ in absc]
    for calcul in range(nbCalculs):
        Points[calcul] = esperanceDuLossInvSensi(n,p,eps,nbEssais)
    p1, = plt.plot(absc, Points)
    p2, = plt.plot(absc, borneInf)
    plt.legend([p1,p2], ["Expected loss", "Borne inférieure donnée par le théorème 1, avec k = 1"])
    plt.show()

# traceBorneInfEsperanceInvSensi(100, 0.5, 1, 100, 100)


'''
OPTIMALITE DU MECANISME D'INVERSE SENSITIVITY
_____________________________________________

Toujours dans le cas d'un n-uplet de B(p), on cherche à montrer que l'expected loss du mécanisme d'inverse 
sensitivity est de l'ordre de 1/n (ce qui impliquera que ce mécanisme est optimal).

Fonction traçant cet expected loss pour le comparer à cet ordre:
en entrée:  - N le n maximal pour lequel on calcule l'expected loss
            - les autres arguments comme précédemment
'''
def traceExpectedLossInvSensi(N, p, eps, nbEssais):
    vect_n = range(10, N+1)
    expectedLosses = [0 for n in vect_n]
    borneInf = [1/ (2*n*(np.exp(2 * eps)+1)) for n in vect_n]
    #équivalent de l'expected loss trouvé par le calcul
    equivalent = [4 * (np.exp(2*eps+1)) / ((1 - np.exp(-eps)) * (2*n*(np.exp(2 * eps)+1))) for n in vect_n]
    for n in range(10, N+1):
        expectedLosses[n-10] = esperanceDuLossInvSensi(n,p,eps,nbEssais)
    p1, = plt.plot(vect_n, expectedLosses)
    p3, = plt.plot(vect_n, borneInf)
    p2, = plt.plot(vect_n, equivalent)
    plt.legend([p1,p2, p3], ["Expected loss", "Equivalent en O(1/n)", "Borne inférieure maximale (k = 1) donnée par le théorème 1"])
    plt.show()

# traceExpectedLossInvSensi(500, 0.5, 1, 100)



'''
CONVERGENCE DES MECANISMES D'INVERSE SENSITIVITY
________________________________________________

Simulation pour vérifier nos résultats sur la convergence des fonctions caractéristiques des Minv,n.
en entrée:  - N la valeur maximale de n
            - p, t, eps
en sortie: affichage de |Phi_Yn (t) - Phi_Y (t)| en fonction de n (censé tendre vers 0)
'''
def verifLevy(N,p, t,eps):
    absc = range(4000, N+1)
    #|Phi_Yn (t) - Phi_Y (t)| 
    distance = [0 for _ in absc]
    # Phi_Y (t)
    PhiY = np.exp(1j*t*p)
    for n in range(4000, N+1):
        X = rd.binomial(1,p,size=n)
        fX = sum(X)/n
        PhiYn = 0
        for k in range(n+1):
            PhiYn += (np.exp(1j * t * k / n - eps * np.abs(n*fX - k) / 2)) / (np.exp(-n*eps*fX/2) - np.exp(eps/2) -1 + np.exp(n*eps*(fX-1)/2))
        #on multiplie par le dénominateur du coefficient de normalisation
        PhiYn *= (1 - np.exp(eps/2))
        distance[n-4000] = np.abs(PhiYn - PhiY)
    plt.plot(absc, distance)
    plt.show()


# verifLevy(5000, 0.5, -22, 1)

