import numpy as np
import math
import numpy.random as rd
import matplotlib.pyplot as plt
import time
import Algorithmes_de_calcul_de_quantiles_de_Google_research


'''
____________________________________________________________________________________________________________________________________

Ce code permet de tracer les estimations, par différents mécanismes exponentiels détaillés dans la partie 4 du rapport, 
des quantiles demandés de V.A. (soit données, soit simulées).
Il produit les figures apparaissant dans la partie 4 du rapport.
____________________________________________________________________________________________________________________________________
'''


'''
Fonction calculant le quantile d'ordre p de manière non bruitée.
en entrée:  - Xi tableau de n VA iid
            - p l'ordre du quantile recherché
en sortie: le quantile d'ordre p
'''
def quantile(Xi,p):
    donnees = np.sort(Xi)
    n = len(donnees)
    if p==0:
        return(-np.inf)
    if p==1:
        return donnees[n-1]
    pos_decimal = p * (n+1)
    pos = int(pos_decimal)
    quant = donnees[pos-1] + (pos_decimal - pos)* (donnees[pos] - donnees[pos-1])
    return quant

#print(quantile([0,1,2,3], 0.75))

'''
Fonction pour calculer plusieurs quantiles sans DP
en entrée:  - X tableau de n VA iid
            - p les ordres des quantiles recherchés
en sortie: les quantiles d'ordres recherchés
'''
def multiQuantiles(X,p):
    m = len(p)
    q = [0 for _ in range(m)]
    for iter in range(m):
        q[iter] = quantile(X, p[iter])
    return q

'''
Algorithme QExp.
en entrée:  -données X = (X1, ..., X_(n))
            - p l'ordre du quantile recherché
            - eps le degré de privacy
            - Lambda bounding parameter
en sortie : quantile d'ordre p, calculé avec eps-DP selon l'algorithme proposé par [Smith, 2011] (cité chez Clément Lalanne)
'''
def QExp(X, p,eps, Lambda):
    Z = np.sort(X)
    k = len(Z)
    for Zi in Z:
        if Zi <0:
            Zi = 0
        if Zi > Lambda:
            Zi = Lambda
    Z1 = [*[0], *Z, *[Lambda]]
    Y = [0 for i in range(k+1)]
    for i in range(k+1):
        Y[i] = (Z1[i+1] - Z1[i])*np.exp(-eps * np.abs(i - p * k))
    #on veut maintenant produire un i avec proba yi/sum(Y).
    Y = [y/sum(Y) for y in Y]
    for i in range(1,k+1):
        Y[i] += Y[i-1]
    proba = rd.random()
    #à la recherche du i désigné par le sort...
    for i in range(k+1):
        if Y[i] >= proba:
            i0 = i
            break
    # "output a uniform draw from Z_(i+1) - Zi"
    return(Z1[i0] + (Z1[i0+1] - Z1[i0]) * rd.random())


#print(QExp([0,50,70,95], 0.5, 0.8, 100))
#59.22261510602171


'''
Traçage des résultats de plusieurs simulations sur des données aléatoires

en entrée:  - n nombre de données à tirer aléatoirement
            - a et b délimitant l'intervalle dans lequel sont ces données
              (rq: b sera pris comme bounding parameter)
            - p
            - eps 
            - N nombre de simulations

'''
def QExpSimu(n,a,b,p,eps,N):
    X = [a + (b-a) * rd.random() for _ in range(n)]
    X.sort()
    #vecteur des quantiles
    Q = [QExp(X,p,eps,b) for _ in range(N)]

    #calcul du quantile sans differential privacy
    vraiQuantile = quantile(X,p)

    #traçage du véritable quantile
    p1 = plt.axvline(x = vraiQuantile, color = 'red', linestyle = '--', alpha = 1)

    #traçage des quantiles avec eps-DP
    countsQ, binsQ = np.histogram(Q)
    p2 = plt.stairs(countsQ, binsQ)
    
    #traçage de l'histogramme
    counts, bins = np.histogram(X)
    p3 = plt.stairs(counts, bins)
    plt.legend([p1, p2, p3], ["Quantile non bruit\'e", "Estimations du quantile avec DP", "Donn\'ees"])
    plt.show()

#QExpSimu(1000, 0, 200, 0.5, 0.05, 500)

'''
Même fonction, mais qui prend en entrée le vecteur X déjà existant et qui calcule le eps tel que 
la probabilité que l'erreur soit plus grande que (b-a)/10 soit inférieure à 0.01 
(où a~min X et b ~ max X).
((b-a)/10 est la taille d'un pas de l'histogramme.)

en entrée:  - X
            - p
            - N nombre de simulations

'''
def QExpSimuIntelligent(X,p,N):
    n = len(X)
    #on enlève 1 au min et ajoute 1 au max pour s'assurer que ces valeurs sont bien prises en compte 
    #lors du calcul de quantile
    a = min(X) - 1
    b = max(X) + 1
    X.sort()
    DeltaListe = [X[i+1] - X[i] for i in range(n-1)]
    for i in range(n-1):
        if DeltaListe[i] == 0:
            DeltaListe[i] = b-a
    Delta = min(DeltaListe)
    #calcul de eps
    eps = 20 * (np.log((b-a)/Delta) + np.log(100)) / (b-a)
    print("eps = ", eps)
    #vecteur des quantiles
    Q = [QExp(X,p,eps,b) for _ in range(N)]

    #calcul du quantile sans differential privacy
    vraiQuantile = quantile(X,p)

    #traçage du véritable quantile
    p1 = plt.axvline(x = vraiQuantile, color = 'red', linestyle = '--', alpha = 1)

    #traçage des quantiles avec eps-DP
    countsQ, binsQ = np.histogram(Q)
    p2 = plt.stairs(countsQ, binsQ)
    
    #traçage de l'histogramme
    counts, bins = np.histogram(X)
    p3 = plt.stairs(counts, bins)
    plt.legend([p1, p2, p3], ["Quantile non bruit\'e", "Estimations du quantile avec DP", "Donn\'ees"])
    plt.show()


X = [rd.randint(200) for _ in range(1000)]
#X = [200 * rd.random() for _ in range(1000)]
#QExpSimuIntelligent(X, 0.5, 500)
#eps =  0.9859179198056883



'''
Traçage des risques quadratiques pour QExp et pour un équivalent avec bruit de Laplace, en fonction de eps / n / Lambda (au choix), 
avec X aléatoire.
en entrée:  - p
            - eps
            - n
            - Lambda
    (rq: selon ce qu'on veut en abscisse, un de ces trois derniers n'est pas pris en abscisse)
            - N nombre de simulations pour chaque risque quadratique
'''

#fonction auxiliaire: mécanisme de Laplace pour le calcul d'un quantile
#(on aurait aussi pu s'inspirer de l'algorithme 3 de "Bounded Space Differentially Private Quantiles")
def QLap(X,p, eps, Lambda):
    theta = quantile(X,p)
    return (theta + rd.laplace(loc=0.0, scale = Lambda / eps, size=None))


def traceEtCompareRisquesQuadrDeEps(p, n, Lambda, N):
    Eps = [i for i in range(1,101)]
    RisqueQuadrExp = [0 for _ in range(100)]
    RisqueQuadrLap = [0 for _ in range(100)]
    for eps in Eps:
        X = [rd.randint(Lambda) for _ in range(n)]
        theta = quantile(X,p)
        for j in range(N):
            EstimExp = QExp(X,p,eps/100, Lambda)
            RisqueQuadrExp[eps-1] += (EstimExp - theta)**2
            EstimLap = QLap(X,p,eps/100, Lambda)
            RisqueQuadrLap[eps-1] += (EstimLap - theta)**2
        RisqueQuadrExp[eps-1] = RisqueQuadrExp[eps-1] / N
        RisqueQuadrLap[eps-1] = RisqueQuadrLap[eps-1] / N
    Eps = [eps/100 for eps in Eps]
    p1, = plt.plot(Eps,RisqueQuadrExp)
    p2, = plt.plot(Eps, RisqueQuadrLap)
    plt.legend([p1,p2], ["Risque quadratique avec bruit exponentiel en fonction de epsilon", "Risque quadratique avec bruit de Laplace en fonction de epsilon"])
    plt.show()

# traceEtCompareRisquesQuadrDeEps(0.25, 10, 200, 150)

def traceEtCompareRisquesQuadrDe_n(p, eps, Lambda, N):
    Vect_n = [10 * i for i in range(1,101)]
    RisqueQuadrExp = [0 for _ in range(100)]
    RisqueQuadrLap = [0 for _ in range(100)]
    for n in Vect_n:
        X = [rd.randint(Lambda) for _ in range(n)]
        theta = quantile(X,p)
        for j in range(N):
            EstimExp = QExp(X,p,eps, Lambda)
            RisqueQuadrExp[n//10 - 1] += (EstimExp - theta)**2
            EstimLap = QLap(X,p,eps, Lambda)
            RisqueQuadrLap[n//10 - 1] += (EstimLap - theta)**2
        RisqueQuadrExp[n//10 - 1] = RisqueQuadrExp[n//10 - 1] / N
        RisqueQuadrLap[n//10 - 1] = RisqueQuadrLap[n//10 - 1] / N
    p1, = plt.plot(Vect_n,RisqueQuadrExp)
    p2, = plt.plot(Vect_n, RisqueQuadrLap)
    plt.legend([p1,p2], ["Risque quadratique avec bruit exponentiel en fonction du nombre de données", "Risque quadratique avec bruit de Laplace en fonction du nombre de données"])
    plt.show()

#en utilisant la simulation précédente, on prend eps = 0.2, pour lequel il y a peu de différences entre Exp et Lap,
#mais un peu quand même.

#traceEtCompareRisquesQuadrDe_n(0.25, 0.2, 200, 150)



# Même fonction, mais avec Lambda en abscisse
def traceEtCompareRisquesQuadrDeLambda(p, eps, n, N):
    VectLambda = [2 * i for i in range(1,101)]
    RisqueQuadrExp = [0 for _ in range(100)]
    RisqueQuadrLap = [0 for _ in range(100)]
    for Lambda in VectLambda:
        X = [rd.randint(Lambda) for _ in range(n)]
        theta = quantile(X,p)
        for j in range(N):
            EstimExp = QExp(X,p,eps, Lambda)
            RisqueQuadrExp[Lambda//2 - 1] += (EstimExp - theta)**2
            EstimLap = QLap(X,p,eps, Lambda)
            RisqueQuadrLap[Lambda//2 - 1] += (EstimLap - theta)**2
        RisqueQuadrExp[Lambda//2 - 1] = RisqueQuadrExp[Lambda//2 - 1] / N
        RisqueQuadrLap[Lambda//2 - 1] = RisqueQuadrLap[Lambda//2 - 1] / N
    p1, = plt.plot(VectLambda,RisqueQuadrExp)
    p2, = plt.plot(VectLambda, RisqueQuadrLap)
    plt.legend([p1,p2], ["Risque quadratique avec bruit exponentiel en fonction du bounding parameter des données", "Risque quadratique avec bruit de Laplace en fonction du bounding parameter des données"])
    plt.show()

#traceEtCompareRisquesQuadrDeLambda(0.25, 0.2, 1000, 150)



'''
en entrée:  - X les données
            - p vecteur (/!\) des m ordres des quantiles recherchés
            -eps degré de privacy
            - Lambda bounding parameter
en sortie: le vecteur des m quantiles recherchés
'''
def IndExp(X, p, eps, Lambda):
    m = len(p)
    Q = [0 for i in range(m)]
    for i in range(m):
        Q[i] = QExp(X, p[i], eps, Lambda)
    return Q

#print(IndExp([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.25, 0.5, 0.75], 0.8, 1))
#[np.float64(0.28414519601546717), np.float64(0.4049300876491281), np.float64(0.6251673710833014)]





'''
RecExp

fonction récursive auxiliaire
en entrée:  - X, p comme précédemment
            - epsQExp tel que QExp soit eps-DP (attention, ce n'est pas le eps tel que RecExp soit eps-DP!)
            - a, b comme précédemment
'''
def auxRecExp(X,p,epsQExp,a,b):
    m = len(p)
    if m == 0:
        return []
    elif m == 1:
        #attention, dans le code les indices de p vont de 0 à m-1
        return [QExp(X,p[0],epsQExp, b)]
    mChapeau = math.ceil(m/2)
    ordre_p = p[mChapeau - 1]
    v = QExp(X,ordre_p,epsQExp,b)
    Xl = []
    Xu = []
    for x in X:
        if x < v:
            Xl.append(x)
        elif x > v:
            Xu.append(x)
    Ql = [q / ordre_p for q in p[0:mChapeau - 1]]
    Qu = [(q-ordre_p) / (1 - ordre_p) for q in p[mChapeau:]]
    return([*auxRecExp(Xl, Ql, epsQExp, a, v), *[v], *auxRecExp(Xu, Qu, epsQExp, v, b)])

def RecExp(X,p,eps,a,b):
    return (sorted(auxRecExp(X, p, eps / (np.log2(len(p)) + 1), a, b)))

#X = [rd.randint(200) for _ in range(1000)]
#X.sort()
#print([quantile(X, 0.25), quantile(X, 0.5), quantile(X, 0.75)])
#[np.float64(56.0), np.float64(103.0), np.float64(151.0)]
#print(RecExp(X, [0.25, 0.5, 0.75], 1, 0, 201))
#[np.float64(55.056252595442906), np.float64(103.33234331048743), np.float64(151.644661876131)]


'''
Traçage des risques quadratiques des fonctions de calcul de plusieurs quantiles
Pour le risque quadratique d'une estimation (e[0], ..., e[m-1]]) d'un paramètre theta de R^m, 
on rappelle la définition:
E[(e[0] - theta[0])^2 + ... + (e[m-1] - theta[m-1])^2]

en entrée:  - p le vecteur des ordres des quantiles
            - n
            - a
            - b
            - N le nombre de simulations qu'on veut faire
'''
def traceEtComparePourMultiQuantilesRisquesQuadrDeEps(p, n, a, b, N):
    # Eps = [eps/100 for eps in Eps]
    # OU:
    Eps = [0.1, 0.5, 0.7, 0.8, 0.9, 1, 2]
    m = len(p)
    # X = [rd.randint(a,b) for _ in range(n)]
    # OU:
    X = [rd.random() for _ in range(n)]
    RisqueQuadrJointExp = [0 for _ in range(7)]
    RisqueQuadrRecExp = [0 for _ in range(7)]
    RisqueQuadrIndExp = [0 for _ in range(7)]
    for indEps in range(7):
        theta = multiQuantiles(X,p)
        for j in range(N):
            EstimJointExp = algos_quantiles_de_Google_research.joint_exp(X,a,b,p,Eps[indEps], False)
            EstimRecExp = RecExp(X,p,Eps[indEps], a, b)
            EstimIndExp = IndExp(X,p,Eps[indEps], b)
            for iter in range(m):
                EstimJointExp[iter] = (EstimJointExp[iter] - theta[iter])**2
                EstimRecExp[iter] = (EstimRecExp[iter] - theta[iter])**2
                EstimIndExp[iter] = (EstimIndExp[iter] - theta[iter])**2
            RisqueQuadrJointExp[indEps] += sum(EstimJointExp)
            RisqueQuadrRecExp[indEps] += sum(EstimRecExp)
            RisqueQuadrIndExp[indEps] += sum(EstimIndExp)
        RisqueQuadrJointExp[indEps] = RisqueQuadrJointExp[indEps] / N
        RisqueQuadrRecExp[indEps] = RisqueQuadrRecExp[indEps] / N
        RisqueQuadrIndExp[indEps] = RisqueQuadrIndExp[indEps] / N
    p1, = plt.plot(Eps,RisqueQuadrJointExp)
    p2, = plt.plot(Eps, RisqueQuadrRecExp)
    p3, = plt.plot(Eps, RisqueQuadrIndExp)
    plt.legend([p1,p2, p3], ["Risque quadratique de l'algorithme JointExp en fonction de epsilon", "Risque quadratique de l'algorithme RecExp en fonction de epsilon", "Risque quadratique de l'algorithme IndExp en fonction de epsilon"])
    plt.show()

#traceEtComparePourMultiQuantilesRisquesQuadrDeEps([0.25, 0.5, 0.75], 100, 0, 1, 50)

'''
La même chose mais en fonction du nombre de quantiles
en entrée: mêmes arguments (sauf p) + eps 
Pour chaque m, les quantiles considérés sont ceux d'ordre i/(m+1) (par exemple, si m=3, ce sont les 3 quartiles)
'''
def traceEtComparePourMultiQuantilesRisquesQuadrDuNbQuantiles(n, eps, a, b, N):
    # X = [rd.randint(a,b) for _ in range(n)]
    # X = [rd.random() for _ in range(n)]
    X = [rd.beta(2,2) for _ in range(n)]
    RisqueQuadrRecExp = [0 for _ in range(10)]
    RisqueQuadrIndExp = [0 for _ in range(10)]
    #on prend les m de 10 en 10 pour que soit plus rapide
    for m in range(1, 101, 10):
        p = [i / (m+1) for i in range(1, m+1)]
        theta = multiQuantiles(X,p)
        for j in range(N):
            EstimRecExp = RecExp(X,p,eps, a, b)
            EstimIndExp = IndExp(X,p,eps, b)
            for iter in range(m-1):
                EstimRecExp[iter] = (EstimRecExp[iter] - theta[iter])**2
                EstimIndExp[iter] = (EstimIndExp[iter] - theta[iter])**2
            RisqueQuadrRecExp[(m-1) // 10] += sum(EstimRecExp)
            RisqueQuadrIndExp[(m-1) // 10] += sum(EstimIndExp)
        RisqueQuadrRecExp[(m-1) // 10] = RisqueQuadrRecExp[(m-1) // 10] / N
        RisqueQuadrIndExp[(m-1) // 10] = RisqueQuadrIndExp[(m-1) // 10] / N
    abscisses = range(1,101, 10)
    p2, = plt.plot(abscisses, RisqueQuadrRecExp)
    p3, = plt.plot(abscisses, RisqueQuadrIndExp)
    plt.legend([p2, p3], ["Risque quadratique de l'algorithme RecExp en fonction du nombre de quantiles", "Risque quadratique de l'algorithme IndExp en fonction du nombre de quantiles"])
    plt.show()

# pour un X de loi beta (a=0, b=1):
# traceEtComparePourMultiQuantilesRisquesQuadrDuNbQuantiles(100, 1, 0, 1, 50)


'''
On calculait ici un risque quadratique (E[ ||estimateur - valeur cible||_2 ²]). 
Calculons maintenant E[ ||estimateur - valeur cible||_inf], 
comme dans les figures p.8 de [Lalanne, Garivier, Gribonval]:
'''
def traceEtComparePourMultiQuantilesRisquesQuadrInfDuNbQuantiles(n, eps, a, b, N):
    # X = [rd.random() for _ in range(n)]
    X = [rd.beta(2,2) for _ in range(n)]
    RisqueQuadrRecExp = [0 for _ in range(10)]
    RisqueQuadrIndExp = [0 for _ in range(10)]
    #on prend les m de 10 en 10 pour que soit plus rapide
    for m in range(1, 101, 10):
        p = [i / (m+1) for i in range(1, m+1)]
        theta = multiQuantiles(X,p)
        for j in range(N):
            EstimRecExp = RecExp(X,p,eps, a, b)
            EstimIndExp = IndExp(X,p,eps, b)
            RisqueQuadrRecExp[(m-1) // 10] += max(EstimRecExp)
            RisqueQuadrIndExp[(m-1) // 10] += max(EstimIndExp)
        RisqueQuadrRecExp[(m-1) // 10] = RisqueQuadrRecExp[(m-1) // 10] / N
        RisqueQuadrIndExp[(m-1) // 10] = RisqueQuadrIndExp[(m-1) // 10] / N
    abscisses = range(1,101, 10)
    p2, = plt.plot(abscisses, RisqueQuadrRecExp)
    p3, = plt.plot(abscisses, RisqueQuadrIndExp)
    plt.legend([p2, p3], ["E[ ||estimateur - valeur cible||_inf] pour l'algorithme RecExp en fonction du nombre de quantiles", "E[ ||estimateur - valeur cible||_inf] pour l'algorithme IndExp en fonction du nombre de quantiles"])
    plt.show()


# traceEtComparePourMultiQuantilesRisquesQuadrInfDuNbQuantiles(100, 1, 0, 1, 50)


