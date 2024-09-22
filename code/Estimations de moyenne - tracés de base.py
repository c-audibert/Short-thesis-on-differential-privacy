import math as m
import numpy.random as rd
import matplotlib.pyplot as plt

'''
____________________________________________________________________________________________________________________________________

Ce code permet de tracer les histogrammes des estimations, par mécanisme gaussien ou de Laplace, de l'espérance de n V.A.
Il produit les figures apparaissant dans la partie 3 du rapport.
____________________________________________________________________________________________________________________________________
'''

'''
Fonction produisant un ensemble de données Xi de taille n puis estimant son espérance par mécanisme de Laplace.
en entrée:  - a, b bornes de l'intervalle dans lequel les données prennent leurs valeurs
            - n le nombre de données
            - eps le degré de privacy
en sortie: l'estimateur simulé de l'espérance
'''
def simuEstimateurMuLap(a,b,n,eps):
    if b<a:
        print("Error: we must have a <= b")
        return
    estimateur = 0
    #avec des données suivant une loi uniforme
    #Xi = n*[a] + (b-a) * rd.random(n)
    #avec des données suivant une loi gaussienne centrée au milieu de [a,b]
    Xi = [rd.normal(loc=(a+b) / 2, scale=(a+b) / 8, size=None) for i in range(n)]
    estimateur = sum(Xi)/n
    print("moy = ", estimateur)
    #avec une neighboring relation en replacement setup:
    DeltaF = (b-a) / n
    estimateur += rd.laplace(loc=0.0, scale = DeltaF / eps, size=None)
    print("mu = ", a+ (b-a) / 2, "and estimateur = ", estimateur)
    return estimateur


#simuEstimateurMuLap(0,1000,1000, 0.8)



'''
Même fonction, mais avec un mécanisme gaussien.
en entrée:  - a, b bornes de l'intervalle dans lequel les données prennent leurs valeurs
            - n le nombre de données
            - eps, delta tels que le mécanisme soit (eps, delta)-differentially private
            - c tel que décrit dans "The Algorithmic Foundations of Differential Privacy" de Dwork et Roth (utile pour minorer 
            l'écart-type du bruit gaussien, de manière à ce que le mécanisme soit (eps, delta)-differentially private).
en sortie: l'estimateur simulé de l'espérance

'''
def simuEstimateurMuGauss(a,b,n,eps,delta,c):
    if b<a:
        print("Error: we must have a <= b")
        return
    if c**2 <= 2*m.log(1.25/delta):
        print("Error: we must have c^2 > 2 ln(1.25 / delta) to ensure (eps,delta) - DP")
        return
    estimateur = 0
    #avec une loi uniforme
    Xi = n*[a] + (b-a) * rd.random(n)
    #avec une loi gaussienne centrée au milieu de [a,b]
    #Xi = [rd.normal(loc=(a+b) / 2, scale=(a+b) / 8, size=None) for i in range(n)]
    estimateur = sum(Xi)/n
    print("moy = ", estimateur)
    #avec une neighboring relation en replacement setup: 
    # Rq1: Delta2(f) = Delta1(f) car les Xi sont à valeurs dans un espace de dimension 1. 
    #Rq2: pour que Deltai(f) = (b-a)/n, on présuppose que les Xi sont à valeurs dans [a,b]. Càd que si les Xi sont gaussiennes par ex,
    #on sq [a,b] est suffisamment grand pour que les Xi n'en sortent pas.
    Delta2F =(b-a) / n
    ecartType = c * Delta2F / eps
    estimateur += rd.normal(loc=0.0, scale=ecartType, size=None)
    print("mu = ", a+ (b-a) / 2, "and estimateur = ", estimateur)
    return estimateur

#on prend delta << 1/n (recommandé dans "The Algorithmic Foundations of Differential Privacy" après la définition 2.4)
#simuEstimateurMuGauss(0,1000,1000, 0.8, 0.00001,5)

'''
Fonction traçant l'histogramme des estimations, par mécanisme de Laplace, de l'espérance de n V.A.
en entrée:  -a, b bornes de l'intervalle
            -n nombre de Xi
            -eps le degré de privacy
            -precision la taille des intervalles de discrétisation des estimations : on considère que les estimations lui appartenant
             sont égales à son inf
            -N le nombre d'estimations réalisées
en sortie: affiche l'histogramme des estimations (à precision près), en fonction des estimations e
Rq: on ne tient compte que des estimations appartenant à [a,b].
    Il faut donc prendre un [a,b] assez grand, sinon le nombre d'estimations risque d'être faible.
'''
def nbOccurrencesEstimationLap(a,b,n,eps,precision,N):
    mu = (a+b) / 2
    #nombre d'intervalles de discrétisation de l'estimation
    nbPasEstimation = round((b-a) / precision)

    x = [a + i * precision for i in range(nbPasEstimation)]
    #le vecteur des ordonnées pour la fonction plot (nombre d'occurrences pour chaque intervalle de discrétisation)
    nbOccurrences = [0 for x in range(nbPasEstimation)]
    for e in range(N):
        estimateur = simuEstimateurMuLap(a,b,n,eps)
        if (estimateur > b) or (estimateur < a):
            continue
        numeroIntervalleDiscr = m.floor((estimateur - a) / precision)
        nbOccurrences[numeroIntervalleDiscr] += 1
    plt.plot(x,nbOccurrences)
    plt.xlabel('estimation e')
    plt.ylabel('nombre d\' occurrences de e')
    plt.title('estimations obtenues pour ' + str(n) + ' VA ~ N((a+b) / 2, (a+b) / 8), avec un bruit de distribution de Laplace, \n une precision de ' + str(precision) + ', ' +str(N) + ' simulations et eps = ' + str(eps))
    plt.grid()
    plt.show()

#nbOccurrencesEstimationLap(0,1000,1000,0.8,0.5,2000)

'''
Même fonction, mais avec un mécanisme gaussien.
en entrée:  -a, b bornes de l'intervalle
            -n nombre de Xi
            - eps, delta tels que le mécanisme soit (eps, delta)-differentially private
            - c tel que décrit dans "The Algorithmic Foundations of Differential Privacy" de Dwork et Roth (utile pour minorer 
            l'écart-type du bruit gaussien, de manière à ce que le mécanisme soit (eps, delta)-differentially private).
            -precision la taille des intervalles de discrétisation des estimations : on considère que les estimations lui appartenant
             sont égales à son inf
            -N le nombre d'estimations réalisées
en sortie: cf. précédemment.
'''

def nbOccurrencesEstimationGauss(a,b,n,eps,delta,c,precision,N):
    mu = (a+b) / 2
    #nombre d'intervalles de discrétisation de l'estimation
    nbPasEstimation = round((b-a) / precision)

    x = [a + i * precision for i in range(nbPasEstimation)]
    #le vecteur des ordonnées pour plot (nombre d'occurrences pour chaque intervalle de discrétisation)
    nbOccurrences = [0 for x in range(nbPasEstimation)]
    for e in range(N):
        estimateur = simuEstimateurMuGauss(a,b,n,eps,delta,c)
        if (estimateur > b) or (estimateur < a):
            continue
        numeroIntervalleDiscr = m.floor((estimateur - a) / precision)
        nbOccurrences[numeroIntervalleDiscr] += 1
    plt.plot(x,nbOccurrences)
    plt.xlabel('estimation e')
    plt.ylabel('nombre d\' occurrences de e')
    plt.title('estimations obtenues pour ' + str(n) + ' VA uniformes, avec un bruit gaussien, \n une precision de ' + str(precision) + ', ' +str(N) + ' simulations et eps = ' + str(eps))
    plt.grid()
    plt.show()

# nbOccurrencesEstimationGauss(0,1000,1000,0.8,0.00001,5,0.5,2000)
