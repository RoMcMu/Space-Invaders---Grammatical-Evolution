#Ronan McMullen - 0451657

import os.path
import random
from PIL.Image import NONE
from deap.tools.crossover import cxOnePoint
import gym
import re
import time
import datetime
import psutil
import platform
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

GAME = 'SpaceInvadersDeterministic-v4'
GRAMMAR = 'space_invaders_grammar_type_1.pybnf'
AMOUNT_OF_EXPRESSION_USED = 0

IND = 0

INDIVIDUAL_LENGTH = 75000 # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 150
P_CROSSOVER = 0.8 # probability for crossover
CROSSOVER_TYPE = 'cx_one_point_effective'
P_MUTATION = 0.03   # probability for mutating an individual
MAX_GENERATIONS = 25
HALL_OF_FAME_SIZE = 1
N_RUNS = 1
T_SIZE = 5

MOTHERSHIP_CURB = False
LIFE_PEN = False
CHECK_INV = False

# set the random seed:
RANDOM_SEED = 809#random.randint(1, 1000)
random.seed(RANDOM_SEED)        
 
def g2p_map (individual):

    """This function accepts a variable length bit string genome. It parses the genome in 8-bit codons.
    Each codon is converted to its corresponding decimal value. This decimal value is modded (%) by the number of choices
    available in the current non-terminal and the resulting value is used to determine which choice is selected."""

    choices = list()
    codon = ""
    i=0
    for bit in individual:
        codon += str(bit)
        i+=1
        if i % 8 == 0:
            n = parse_codon(codon)
            choices.append(n)
            codon = ""

    phenome = prod_rule_dict['S'][0]
    print_out = False
    i=0
    
    while(True):

        non_terminal = re.search("<.*?>", phenome)

        if(non_terminal is not None and i < len(choices)):
            
            component = non_terminal.group(0)
            rule = prod_rule_dict[component]
            phenome = phenome.replace(component, rule[choices[i] % len(rule)], 1)

            if(print_out):
                print("Phenome length:", len(phenome))
                print("Component:", component)
                print("Rule:", rule, ", len(rule):", len(rule))
                print("Codon to decimal element",  i, "=", choices[i])
                print(choices[i], "%", len(rule), ":", choices[i] % len(rule))
                print("Choice:", rule[choices[i] % len(rule)] )
                print("Phenome =", phenome)
                print(" ")
        else:
            break
        i+=1
   
    return phenome    
    
def bnf_parse (filename):
    """this function will parse the specified bnf file
    and add it to a usable data structure that can be accessed
    by the g2p_map function.
    The bnf grammar is kept in a seperate accessible .pybnf file"""

    if not os.path.isfile(filename):
        print('File', filename, 'does not exist.')
    else:
        with open(filename) as f:
            grammar = f.read().splitlines()

        for g in grammar:
            g = g.replace(" ", "")
            rule = g.split("::=")
            title = rule[0]
            components = rule[1].split("|")    
            prod_rule_dict[title] = components
         
def parse_codon (codon):
    """this helper function parses an 8bit binary number
    and returns its decimal representation"""

    c = list(codon)
    n = 0
    i = 7

    for bit in c:
        if bit == "1":
            n = n + pow(2,i)
        i -= 1

    return n

def fitness_eval (phenome):

    global AMOUNT_OF_EXPRESSION_USED
    global IND
    global MOTHERSHIP_CURB
    global LIFE_PEN
    global CHECK_INV

    life_bool = bool

    phenome.strip()
    phenome_list = phenome.split("-")
    del phenome_list[-1]

    int_list = [int(x) for x in phenome_list]
    expression = []

    #at this point the phenome will be comprised, alternately of an action and a number corresponding
    #to how many times that action should be repeated.
    #The following section parses the phenome and copies the repeated action the specified number of times.
    i=0; j=1
    for _ in range((len(int_list)//2)):

        for _ in range(int_list[j]):
            expression.append(int_list[i])
        i += 2
        j += 2
    
    env = gym.make(GAME)    
    
    env.reset()
    fitness = 0

    actions_taken=0
    is_done = False

    action = 0
    frame = NONE

    life_penalty = 0

    IND += 1

    while not is_done:
          
        if (actions_taken < len(expression)):
            
            #carry out appropriate function or assign action based on the current value in the expression.

            if(expression[actions_taken] == 6):
                if frame is not NONE:
                    action = check_bounding_box(frame)
                else:
                    action = 0
            elif(expression[actions_taken] == 7):
                if frame is not NONE:
                    action = seek_lowest_invader(frame, True)
                else:
                    action = 0
            elif(expression[actions_taken] == 8):
                if frame is not NONE:
                    action = seek_lowest_invader(frame, False)
                else:
                    action = 0
            elif(expression[actions_taken] == 9):
                if frame is not NONE:
                    action = seek_side_invader(frame, True)
                else:
                    action = 0
            elif(expression[actions_taken] == 10):
                if frame is not NONE:
                    action = seek_side_invader(frame, False)
                else:
                    action = 0        
            else:
                action = expression[actions_taken]  
        else:
            print("No actions left, ending game.")
            print("actions taken:", actions_taken, "len expression:", len(expression))
            break
        actions_taken += 1

        frame, reward, is_done, _ = env.step(action)
        
        frame = process_frame(frame)


        if(MOTHERSHIP_CURB == True and reward>199):
                reward = 50
   
        fitness += reward

        penalty_amount = 6

        if(CHECK_INV == True and check_invasion(frame)):
                fitness -= (penalty_amount*20)

        if(LIFE_PEN == True):
            
            life_on_screen = check_life(frame)

            if life_on_screen == True:
                life_bool = True

            if life_bool == True and life_on_screen == False:
                r = fitness/penalty_amount
                if(r == 0.0):
                    r = 50

                life_penalty += (1000 * (10/r))
                life_bool = False
        
        env.render()
    env.close()

    if(LIFE_PEN == True and life_penalty == 0):
        #in the event that space invaders hit the ground and the game ends. Need to put in a check for this.
        life_penalty = 500
    
    fitness = fitness - life_penalty

    if(actions_taken/len(expression) > AMOUNT_OF_EXPRESSION_USED):
        AMOUNT_OF_EXPRESSION_USED = actions_taken/len(expression)
    
    fitness = (fitness,)

    return fitness

def check_life(frame):

    f = frame[188:193,63:75] #this corresponds to the area of the screen that the number of lives appears in.

    life_top_corner = frame[33:34,18:19] #a part of the screen that is common to each of the three life numbers, (3,2,1)

    last_f = np.empty(f.shape)
 
    if (f.max() == 0.5) & (life_top_corner != 0.4):#then there is a number on screen and its not the start of a game. Ergo, a life has been lost.
        
        if(last_f is not NONE and np.allclose(f, last_f)):
            
            return False

        last_f = np.copy(f) 
        return True

    else:

        return False

def check_invasion(frame):
    f = frame[192:195,0:140] #this corresponds to strip of pixels at ground level. 3 deep.
    a = np.where(f==0.4)

    if(len(a[0]) != 0):
        return True
    else:
        return False

def process_frame(frame, downsample=False, downsample_amnt=1):

    #Trim frame to relevant area
    frame = frame[0:195, 20:140]

    #change pixel values to between 0-1  
    normalised_frame = frame/255

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grey_image = np.dot(normalised_frame, rgb_weights)
    
    #PIXEL VALUES:
    #Backgorund                     = 0.0
    #Defender                       = 0.38481961
    #Defender_and_Invader Bullets   = 0.55680706
    #Invader                        = 0.47849647
    #Bunker                         = 0.42110549
    #Life number                    = 0.52338745

    grey_image[(grey_image > .38) & (grey_image < .39)] = 0.2 #defender
    grey_image[(grey_image > .42) & (grey_image < .43)] = 0.3 #bunker
    grey_image[(grey_image > .47) & (grey_image < .48)] = 0.4 #invaders
    grey_image[(grey_image > .50) & (grey_image < .53)] = 0.5 #life indicator
    grey_image[(grey_image > .55) & (grey_image < .56)] = 1.0 #bullets

    return grey_image

def get_position(processed_frame):
    #position is in row 185, 
    #It will be an integer that represents which column of the row the tip of the defender is in
    position = np.where(processed_frame[185] == 0.2)
    if(len(position[0]) != 0):
        return position[0]
    else:
        return NONE

def check_bounding_box(frame):
    
    location = get_position(frame)
 
    """location is an int that corresponds to the position on row 185 of the tip of the defender.
    this function constructs a bounding box around this point and checks to see if there is an 
    occurence of a bullet, moving downwards within it. Bullet = 1.0"""
    if(location is not NONE): #checks to see if the defender is on screen or not (could be flashing)

        width = 4
        b_box = frame[178:193,location[0] - (width) :location[0]+(width+1)]

        if(1.0 in b_box):

            bullet_position = np.where(b_box == 1.0)
            x = ((bullet_position[1][0] - width),width)  

            #if x[0] < 0 then the bullet is on the left so move right if possible
            #if x[0] == 0 then the bullet is in middle so move whichever way possible
            #if x[0] > 0 then the bullet is on the right so move left if possible   
             
            # action 4 is move and shoot left.
            # action 5 is move and shoot right.            
            
            if x[0] < 0:
                #94 corresponds to the rightmost column the tip of the defender can be in.
                if location[0] > 94:
                    action = 5

                else:
                    action = 4
            
            if x[0] > 0:
                #17 corresponds to the leftmost column the tip of the defender can be in.
                if location[0] < 17:
                    action = 4

                else:
                    action = 5
            
            if x[0] == 0:

                if location[0] < 17:
                    action = 4

                elif location[0] > 94:
                    action = 5

                else:
                    if location[0] < 57:
                        action = 4

                    else:
                        action = 5

        else:
            action = 1 #fire
    else:  
        action = 0     

    return action

def seek_lowest_invader(processed_frame, side):

    invader_value = 0.4
    result = np.where(processed_frame == invader_value)

    action = 0
    if len(result[0] != 0):

        row = result[0].max()
        possible_targets = np.where(processed_frame[row] == invader_value)

        #possible_targets[0] min() is high, max() is low (row)
        #possible_targets[1] min() is left, max() is right (columns)

        location = get_position(processed_frame)
        

        if side == True:
            target = possible_targets[0].max() #Seek rightmost lowest
            
        else:
            target = possible_targets[0].min() #Seek leftmost lowest
    
        if location is not NONE:
            if location[0] < target:
                action = 4 #move right, toward target column
            elif location[0] > target:
                action = 5 #move left, toward targer column
            else:
                action = 1 #shoot

            
    return action

def seek_side_invader(frame, side):

    invader_value = 0.4
    result = np.where(frame == invader_value)
    
    action = 0
    if len(result[0] != 0):
        if(side == True):
            target = result[1].max()
        else:
            target = result[1].min()
        
        location = get_position(frame)

        if location is not NONE:
            if location[0] < target:
                action = 4 #move right, toward target column
            elif location[0] > target:
                action = 5
            else:
                action = 1

    return action
        
def return_fitness(individual):
    phenome = g2p_map(individual)
    fitness= fitness_eval(phenome)
    return fitness

def cx_one_point_effective(ind1, ind2):

    size = min(len(ind1), len(ind2))
    size = int(size*AMOUNT_OF_EXPRESSION_USED)
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2

def cx_two_point_effective(ind1, ind2):

    size = min(len(ind1), len(ind2))
    size = int(size*AMOUNT_OF_EXPRESSION_USED)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2

toolbox = base.Toolbox()

production_rules = []
prod_rule_dict = {}

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, INDIVIDUAL_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

toolbox.register("evaluate", return_fitness)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=T_SIZE)

# crossover:
toolbox.register("mate", cx_one_point_effective)

# mutation:
toolbox.register("mutate", tools.mutFlipBit, indpb=P_MUTATION)


def main():
    bnf_parse("./grammars/"+GRAMMAR)

    global N_RUNS
    num_runs = 0

    max_list = []
    avg_list = []
    min_list = []
    std_list = []

    for _ in range(0, N_RUNS):
        
        num_runs +=1

        start = datetime.datetime.now()
        # create initial population (generation 0):
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # define the hall-of-fame object:
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)

        # perform the Genetic Algorithm flow:
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                  ngen=MAX_GENERATIONS,
                                                  stats=stats, halloffame=hof, verbose=True)

        # Genetic Algorithm is done - extract statistics:
        max_fitness_values, mean_fitness_values = logbook.select("max", "avg")

        end = datetime.datetime.now()
        runtime = end - start

        details = ('Game: ' + GAME  +  
                   '\nGrammar: ' + GRAMMAR + 
                   '\nSeed: ' + str(RANDOM_SEED) + 
                   '\nPop: ' + str(POPULATION_SIZE) + 
                   '\nGenerations: ' + str(MAX_GENERATIONS) + 
                   '\nIndividual Length: ' + str(INDIVIDUAL_LENGTH) + 
                   '\ncx: ' + str(P_CROSSOVER) + 
                   '\nCrossover type: ' + CROSSOVER_TYPE + 
                   '\nmut: ' + str(P_MUTATION) + 
                   '\n\nRuntime: ' + str(runtime) + 
                   '\nMothership Curb: ' + str(MOTHERSHIP_CURB) +  
                   '\nLife Penalty: ' + str(LIFE_PEN) +  
                   '\nInvasion Check: ' + str(CHECK_INV) + '\n')
    
        cpufreq = psutil.cpu_freq()
        
        cpu_info = (platform.processor() + 
                   '\nPhysical cores: ' + 
                    str(psutil.cpu_count(logical=False)) +
                   '\nTotal cores' + str(psutil.cpu_count(logical=True)) + 
                   '\nMax Frequency: ' + str(cpufreq.max) + 'Mhz\n')

        stamp = time.strftime("%d_%b_%Y-%H_%M")
        file_name = GAME + '_' + stamp

        # Genetic Algorithm is done with this run - extract statistics:
        mean_fitness_values, std_fitness_values, min_fitness_values, max_fitness_values  = logbook.select("avg", "std", "min", "max")
        
        # Save statistics for this run:
        avg_list.append(mean_fitness_values)
        std_list.append(std_fitness_values)
        min_list.append(min_fitness_values)
        max_list.append(max_fitness_values)

        f = open("./HallOfFame/"+file_name, "a")
        f.write("="*40)
        f.write(str(end))
        f.write("="*40)
        data = '\nRun ' + str(num_runs) + ' of ' + str(N_RUNS) 
        f.write(data)
        f.write('\n\n')
        f.write(details)
        f.write('MaxFitness Value: ')
        f.write(str(max(max_fitness_values)))
        f.write('\n\n')
        f.write('\nCPU Info:\n')
        f.write(cpu_info)
        f.write(str(hof.items[0]))
        f.close()

    # Genetic Algorithm is done (all runs) - plot statistics:
    x = np.arange(0, MAX_GENERATIONS+1)
    avg_array = np.array(avg_list)
    std_array = np.array(std_list)
    max_array = np.array(max_list)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness')
    plt.errorbar(x, avg_array.mean(0), yerr=std_array.mean(0),label="Average",color="Red")
    plt.errorbar(x, max_array.mean(0), yerr=max_array.std(0),label="Best", color="Green")
    plt.show()


if __name__ == "__main__":
    main() 


  