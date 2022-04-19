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

GAME = 'SpaceInvaders-v4'
GRAMMAR = 'space_invaders_grammar_type_2.pybnf'

INDIVIDUAL_LENGTH = 4000 # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.8 # probability for crossover
CROSSOVER_TYPE = 'cx_one_point'
P_MUTATION = 0.03   # probability for mutating an individual
MAX_GENERATIONS = 15
HALL_OF_FAME_SIZE = 5
N_RUNS = 1
T_SIZE = 5

CHECK_INV = False

ACTION = 0

# set the random seed:
RANDOM_SEED = random.randint(1, 1000)  
random.seed(RANDOM_SEED)  

def g2p_map (individual):

    """This function accepts a variable length bit string genome. It parses the genome in 8-bit codons.
    Each codon is converted to its corresponding decimal value. This decimal value is modded (%) by the number of choices
    available in the current non-terminal and the resulting value is used to determine which choice is selected."""

    indent_stack = []
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
    print_mapping_process = False
    
    i=0
    indent = 0
    tree_depth = 0

 
    while(True):

        non_terminal = re.search("<.*?>", phenome)

        if(non_terminal is not None and i < len(choices) and tree_depth < 20):

            component = non_terminal.group(0)

            if component == "<indent>":
                #increment the current indent level
                indent += 1

            if component == "<unindent>":
                #decrement the current indent level
                indent -=1
                
            if component == ("<reset_indent>"):
                #reset indent level and tree depth.
                indent = 0
                tree_depth = 0

            if component == ("<push_tree>"):
                indent_stack.append(indent)

            if component == ("<pop_tree>"):
                tree_depth  = indent_stack.pop()

            rule = prod_rule_dict[component]
            choice = rule[choices[i] % len(rule)]

            if(choice.startswith("-I-")):
                #indent to current indent level
                for _ in range(indent):
                    choice = "\t" + choice

            if(choice.startswith("-M-")):
                #match current tree depth
                for _ in range(tree_depth):
                    choice = "\t" + choice
                 
            phenome = phenome.replace(component, choice, 1)

            if(print_mapping_process):
                print("Indent Level:", indent)
                print("Tree depth:", tree_depth)
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
    
    phenome = process_phenome(phenome)

    return phenome   
    
def process_phenome(phenome):
    phenome = phenome.replace("-NL-", "\n")
    phenome = phenome.replace("-X-", "")
    phenome = phenome.replace("-I-", "")
    phenome = phenome.replace("-Q-", "")
    phenome = phenome.replace("-PP-","")
    phenome = phenome.replace("-PH-", "")
    phenome = phenome.replace("-M-", "")
    phenome = phenome.replace("$", "")
    phenome = phenome.replace("-le-", " <= ")
    phenome = phenome.replace("-ge-", " >=")

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

    #This is a penalty applied in the event that an agent manages to shoot the mothership
    #The mothership awards 200 points, as it is usually a chance occurance I felt it inappropriate 
    #to award it so highly.
    mothership_curb = True
    fitness = 0
    is_done = False
    frame = NONE
    syntax_flag = True
    
    env = gym.make(GAME)    
    env.reset()

    while not is_done:
        set_action(0)
        if frame is not NONE:
            check_invader_height(frame)
            try:
                exec(phenome)

            except SyntaxError:
                syntax_flag = False
                set_action(0)

           
        frame, reward, is_done, _ = env.step(ACTION)

        frame = process_frame(frame)
        
        if(mothership_curb == True and reward > 199):
                reward = 50  
        
        fitness += reward

        penalty_amount = 6

        if(CHECK_INV == True and check_invasion(frame)):
                fitness -= (penalty_amount*20)

        env.render()

    env.close()

    if syntax_flag == False:
        fitness = -1

    return fitness,

def set_action(action):
    global ACTION
    ACTION = action

def check_invasion(frame):
    f = frame[192:195,0:140] #this corresponds to strip of pixels at ground level. 3 deep.
    a = np.where(f==0.4)

    if(len(a[0]) != 0):
        return True
    else:
        return False

def check_invader_amount(processed_frame):
    
    invader_value = 0.4

    result = np.where(processed_frame == invader_value)

    amount = (len(result[0])/1374)*100 #1374 total invader pixels
    

    if amount is None:
        amount = 0

    #the value returned is essentially, what percentage of invader pixels remain on the screen.
    return amount

def check_invader_height(frame):
    
    invader_value = 0.4
    result = np.where(frame == invader_value)
    row = None

    if len(result[0] != 0):
        row = result[0].max() # lowest row.
        #195 is ground level, the defender is 10 pixels high so the last chance it has to kill is at row 185.
        #highest possible invader pixel is row 31
        #lowest possible (of any importance) is row 185
        
        row = ((row-31)/185) * 100
        #print(row)
        
    if row is None:
        row = 0
    
    #the value returned is essentially, a percentage of how far the lowest invaders have descended is to the tip of the Defender.
    return row

def check_block_width(frame):

    invader_value = 0.4
    result = np.where(frame == invader_value)
    width = 0

    if len(result[0] != 0):
        width = result[1].max() - result[1].min()
        #maximum width is 87

    width = (width/87) * 100

    #the value returned is a percentage of how wide the block of invaders is compared to the start of the game.
    return width

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

def get_position(frame):

    #position is in row 185. 
    #It will be an integer that represents what column of the row the tip of the defender is in.

    position = np.where(frame[185] == 0.2)

    if(len(position[0]) != 0):
        return position[0]
    else:
        return NONE

def check_bounding_box(processed_frame, height, width):
    
    defender_row = processed_frame[185]
    
    x_y_location_values = np.where(defender_row == 0.2)
    
    if(len(x_y_location_values[0]) != 0):
        location = x_y_location_values[0]
    else:
        location = NONE
   

    """location is an int that corresponds to the position on row 185 of the tip of the defender.
    this function constructs a bounding box around this point and checks to see if there is an 
    occurance of a bullet, moving downwards within it. Bullet = 1.0"""

    if(location is not NONE): #checks to see if the defender is actually on screen or not (could be flashing)

        #row 195 corresponds to the bottom row of pixels of the Defender.
        #the bounding box is by default as tall as the defender 195 - 185, 10 pixels.
        b_box = processed_frame[185-height : 195, location[0] - width : location[0]+(width+1)]

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
            action = seek_lowest_invader(processed_frame, False, False)
    else:
        action = 0
    return action

def seek_lowest_invader(processed_frame, side, picky):

    invader_value = 0.4
    result = np.where(processed_frame == invader_value)

    action = 0
    if len(result[0] != 0):

        row = result[0].max()
        possible_targets = np.where(processed_frame[row] == invader_value)

        #possible_targets[0] min() is high, max() is low (row)
        #possible_targets[1] min() is left, max() is right (columns)

        location = get_position(processed_frame)
        
        if picky == True:
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
                    action = 1 #target is directly above, fire!

        else:
            #if picky is not true then the Defender will always choose the nearest lowest invader.
            if location is not NONE:

                nearest = 1000
                index = 0

                for i in range(len(possible_targets[0])):
                    current_distance = abs(location[0] - possible_targets[0][i])

                    if abs(current_distance < nearest):
                        nearest = abs(current_distance)
                        index = i
                
                if location[0] < possible_targets[0][index]:
                    action = 4 #move right, toward target column
                elif location[0] > possible_targets[0][index]:
                    action = 5 #move left, toward target column
                else:
                    action = 1 #target is directly above, fire!

    return action

def seek_side_invader(processed_frame, side):

    invader_value = 0.4
    possible_targets = np.where(processed_frame == invader_value)
    
    #possible_targets[0] min() is high, max() is low (row)
    #possible_targets[1] min() is left, max() is right (columns)
    action = 0
    if len(possible_targets[0] != 0):

        if(side == True):
            target = possible_targets[1].max()
        else:
            target = possible_targets[1].min()
        
        location = get_position(processed_frame)

        if location is not NONE:
            if location[0] < target:
                action = 4 #move right, toward target column
            elif location[0] > target:
                action = 5 #move left, toward target column
            else:
                action = 1
    return action
  
def camp():
    return 1
    
def return_fitness(individual):
    
    phenome = g2p_map(individual)
    fitness= fitness_eval(phenome)
    
    return fitness

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

# Single-point crossover:
toolbox.register("mate", cxOnePoint)

# Flip-bit mutation:
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

        run_data = '\nRun ' + str(num_runs) + ' of ' + str(N_RUNS) 

        details = ('\nGame: ' + GAME  +  
                   '\nGrammar: ' + GRAMMAR + 
                   '\nSeed: ' + str(RANDOM_SEED) + 
                   '\nPop: ' + str(POPULATION_SIZE) + 
                   '\nGenerations: ' + str(MAX_GENERATIONS) + 
                   '\nIndividual Length: ' + str(INDIVIDUAL_LENGTH) + 
                   '\ncx: ' + str(P_CROSSOVER) + 
                   '\nCrossover type: ' + CROSSOVER_TYPE + 
                   '\nmut: ' + str(P_MUTATION) + 
                   '\n\nRuntime: ' + str(runtime) + '\n')
    
        cpu_freq = psutil.cpu_freq()
        
        cpu_info = (platform.processor() + 
                   '\nPhysical cores: ' + 
                   str(psutil.cpu_count(logical=False)) +
                   '\nTotal cores' + str(psutil.cpu_count(logical=True)) + 
                   "\nMax Frequency: " + 
                   str(cpu_freq.max) + "Mhz\n")

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
        f.write(run_data)
        f.write('\n\n')
        f.write(details)
        f.write('MaxFitness Value: ')
        f.write(str(max(max_fitness_values)))
        f.write('\n\n')
        f.write('\nCPU Info:\n')
        f.write(cpu_info)
        f.write("\nBest Phenotypes:\n")
        for i in range(len(hof.items)):
            f.write(g2p_map(hof.items[i]))
            f.write("\n///////////////////////////////////////////////\n")
        f.write("\nGenotype:\n")    
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