# Import dependencies
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

# Part of python's Data Model
# Objects are python's abstraction of data and way to interact with other data
# Create City class


class City:
    # Double under or dunder methods are pre existing methods in python classes like init
    # These allow you to simulate the behaviour of built in types such as to string or len
    def __init__(self, x, y):  # init function automatically runs on initializing an object in this class and can replace 'self' with others also
        self.x = x
        self.y = y

    # Object method  to be called upon an object of this class
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis**2) + (yDis**2))
        return distance

    # Default print method in python for classes is unsatisfactory
    # Use either repr for object representation or str
    # Default inspection of object calls upon repr and the output is supposed to be unambiguous
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    # str objective is to be readable

    # Create fitness class as inverse of route distance
    class Fitness:
        def __init__(self, route):
            self.route = route
            self.distance = 0
            self.fitness = 0.0

        def routeDistance(self):
            if self.distance == 0:
                pathDistance = 0
                for i in range(0, len(self.route)):
                    fromCity = self.route[i]
                    toCity = None
                    if i+1 < len(self.route):
                        toCity = self.route[i + 1]
                    else:
                        toCity = self.route[0]
                    pathDistance += fromCity.distance(toCity)
                self.distance = pathDistance
            return self.distance

        def routeFitness(self):
            if self.fitness == 0:
                self.fitness = 1/float(self.routeDistance())
            return self.fitness

# Create population which consists of multiple possible routes
# Create one object in the population


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

# Create the initial population


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

# Determine fitness


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        # Creates key pair with i and the fitness score
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    # Items returns tuples from dictionary
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)
    # key needs a function and not value to apply on the inputs, itemgetter return the nth element from the input tuple or list; lambda can work just as well
    # Python treats functions as first class citizens, they can be passed onto other functions as arguments or stored in variables
    # Callable are objects that indicate if something can be called

# Selecting mating pool
# Two ways to do it
# 1. Fitness Proportionate Selection
# Fitness of each individual relative to population is used to assign a probability of selection(roulette wheel)
# 2. Tournament Selection
# A set number of population is randomly selected and the one with the highest fitness is selected as parent and repeated with second set
# Also use elitism which ensure best performing individuals will automatically carry over


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            # iat is to access a single value by integer position, at for single value by label pair
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

# With the ids for mating pool extract the selected individuals


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# Breeding
# Usually its 1/0, but in our case as all cities need to be included we will do an ordered crossover ensuring no repeated cities in routes


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    #child = childP1 + childP2
    # Edit based on comment
    child = childP2[:startGene] + childP1 + childP2[startGene:]
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

# Mutation
# Again its usually random swapping but here it will be swapping two cities randomly


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# Repeat the whole process


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Evolution in motion


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# Running the GA
cityList = []

for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200),
                    y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100,
                 eliteSize=20, mutationRate=0.01, generations=500)
