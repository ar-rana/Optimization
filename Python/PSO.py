#python code for Particle swarm optimization 
import random
import numpy as np
from tkinter import messagebox

#define class particle(define each particle)
class Particle: 
    def __init__(self, position):
        self.position = position 
        self.velocity = np.zeros_like(position) #0
        self.best_positon = position
        self.best_fitness = float('inf') # positive infinity

def PSO(ObjF, Pop_size, D, MaxT): #objective function, population size, dimension, maximum iterations
    swarm_best_position=None
    swarm_best_fitness=float('inf')
    particles=[]

    #position initialization for each particle
    position = np.random.uniform(-0.5,0.5,D) # upper bound, lower bound, dimention (genrating random population with defined LB and UB)
    particle = Particle(position)
    particles.append(particle)

    #fitness value updation
    fitness = ObjF(position)
    if fitness<swarm_best_fitness :
        swarm_best_fitness=fitness
        swarm_best_position=position

        particle.best_position=position
        particle.best_fitness=fitness

    #PSO Main Loop
    for itr in range(MaxT):
        for particle in particles:
            #update velocity
            w = 0.8 #weight inertia
            c1 = 1.2 #coefficient c1 and c2
            c2 = 1.2

            r1 = random.random() # 2 ranodm numbers
            r2 = random.random()

            #formula for velocity calculation
            particle.velocity =(w*particle.velocity+c1*r1*(particle.best_position-particle.position)+c2*r2+(swarm_best_position-particle.position))            
            #new position
            particle.position += particle.velocity

            #evaluate fitness
            fitness = ObjF(particle.position)

            #update personal best fitness value
            if fitness<particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            #update global best
            if fitness<swarm_best_fitness:
                swarm_best_fitness = fitness
                swarm_best_position = particle.position

    return swarm_best_position, swarm_best_fitness
        
#define objective function
def F1(x): #x = position
    return np.sum(x**2)

#we can create more objective functions like here:
def F2(x): 
    return np.max(np.abs(x))
        
Objective_Function = {
    'F1': F1,
    'F2': F2        #add the new objective function here
}

# Parameters
Pop_size = 10
MaxT = 100
D = 2

#Iteration over objective function using PSO
for funcName, ObjF in Objective_Function.items():
    Output = "Running function = " + funcName + "\n"
    best_position, best_fitness = PSO(ObjF, Pop_size, D, MaxT)
    Output += "Best Position " + str(best_position) + "\n"
    Output += "Best Fitness " + str(best_fitness) + "\n"
    Output += "\n"

    messagebox.showinfo("POS RESULT ", Output)