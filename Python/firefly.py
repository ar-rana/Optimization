#firefly algorithm using python
import numpy as np


#Define objective functions
def Objective_Function(x):
    return sum(x**2)

#Define Firefly Algorithm
def Fire_Fly(Obj_Func, LB, UB, Dim, MaxT, Pop_Size, gamma = 1, beta = 2, alpha = 0.2, alpha_damp = 0.98):
    #objective function, lower bound ,upper bound, dimention, maximum iteration, population size, gamma, beta, alpha, alpha_damping
    
    #Initialize Best Solution
    Best_Solution = {"Position": None, "Cost": np.inf} #np.inf = +ve infinity

    #Store Best fitness
    Best_Cost = np.zeros(MaxT)

    #Initialize fireflies population (genrating random population at first with defined bounds LB nad UB)
    fireflies = [{"Position": np.random.uniform(LB,UB,Dim), "Cost": None}for _ in range(Pop_Size)]

    #Calculate fitness values
    for i in range(Pop_Size):
        fireflies[i]["Cost"] = Obj_Func(fireflies[i]["Position"])

    #Check infinite cost
    if (any(not np.isfinite(firefly["Cost"])for firefly in  fireflies)):
        print("Solution have infinite cost")
        return Best_Solution, Best_Cost
    
    #firefly main loop start now
    for it in range(MaxT):
        New_Pop = [{"Position": None, "Cost": None} for _ in  range(Pop_Size)]
        for i in range(Pop_Size):
            for j in range(Pop_Size):
                if (fireflies[j]["Cost"] < fireflies[i]["Cost"]): 
                    r_ij = np.linalg.norm(fireflies[i]["Position"]-fireflies[j]["Position"])  #d_max
                    beta = beta0 * np.exp(-gamma * r_ij ** 2)
                    e = alpha * (np.random.rand(Dim)-0.5) * (UB-LB)

                    New_Sol = {"Position": fireflies[i]["Position"] + beta*(fireflies[j]["Position"] - fireflies[i]["Position"]) + e, "Cost": None}
                    New_Sol["Position"] = np.maximum(New_Sol["Position"], LB)
                    New_Sol["Position"] = np.minimum(New_Sol["Position"], UB)
                    New_Sol["Cost"] = Obj_Func(New_Sol["Position"])

                    if New_Sol["Cost"] < fireflies[i]["Cost"]:
                        fireflies[i] = New_Sol
                        if fireflies[i]["Cost"] < Best_Solution["Cost"]:
                            Best_Solution = fireflies[i].copy()

                        #firefly replacement in new population
                        New_Pop[i] = fireflies[i].copy()

        #Merge population
        pop = sorted([individual for individual in fireflies + New_Pop if individual["Cost"] is not None], key = lambda x: x["Cost"])
        pop = pop[:Pop_Size]

        #sort population 
        fireflies = sorted (pop, key = lambda x: x["Cost"])

        #damp mutation coefficient
        alpha *= alpha_damp

        #store best solution
        Best_Solution[it] = Best_Solution["Cost"]

        #display itration wise solution
        print(f"Iteration {it + 1}: Best Cost = {Best_Cost[it]}")

        #display best solution
        print(f"Best solution at it {it + 1}: {Best_Solution['Position']} (Cost: {Best_Solution['Cost']})")

    return Best_Solution, Best_Cost


#Initialize Parameters
Dim = 3     #dimension
LB = -5     #Lower bound  
UB = 5      #upper bound

#Firefly Algorithm Parameters
#Population Size
Pop_Size = 50

#Maximun Iteration
MaxT = 100

#Light Absorbtion Coefficient
gamma = 1

#Attraction Coefficient
beta0 = 1

#mutation Coefficient
alpha = 0.1

#damping ratio
alpha_damp = 0.95

# run firefly algorithm 
Best_Solution, Best_Cost = Fire_Fly(Objective_Function, LB, UB, Dim, MaxT, Pop_Size, gamma , beta0, alpha, alpha_damp)

#display 
print("\n Best Solution: ", Best_Solution["Position"])
print("\n Best Fitness Value: ", Best_Solution["Cost"])