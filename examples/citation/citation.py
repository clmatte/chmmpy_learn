#
# https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
#
# A simple process detection application using an HMM
#
# We assume the execution of a process with a jobs with fixed lengths.  Jobs may have
# precedence constraints that limit their execution until after a preceding job has completed.
# Additionally, there may be random delays at the beginning of each job.
#
# A simulation is used to generate data from possible executions of the process, and an HMM is
# trained using the simulation data.  Each hidden state in the HMM is associated with a possible
# state of the process, which corresponds to the set of simultaneous executing jobs.
#

import yaml
import random
import pyomo.environ as pe
import numpy as np

from chmmpy import HMMBase
from chmmpy.util import state_similarity, print_differences, run_all


class Citation(HMMBase):
    # Load Citation Description
    # Creates HMM parameters 
    def load_process(self, *, data=None, filename=None):
        #Loads from file
        if filename is not None:
            with open(filename, "r") as INPUT:
                data = yaml.safe_load(INPUT)
        assert data is not None

        alldata = self.data
        alldata.N = data["num_categories"]
        n = alldata.N #I got tired of writing alldata.H over and over again
        alldata.sim = data["sim"]
        alldata.seed = data["sim"].get("seed", None)
        self.name = data["name"]
        alldata.transition_prob = data["transition_prob"]
        alldata.noise_prob = data["noise_prob"]
        
        self.data.N = n
        
        #Creates HMM
        self.start_probs = [1/n]*n
        self.emission_probs = [[1-alldata.noise_prob if i == j else alldata.noise_prob/(n-1) for j in range(n)] for i in range(n)]
        self.trans_mat = [[1-alldata.transition_prob if i == j else alldata.transition_prob/(n-1) for j in range(n)] for i in range(n)]

    #Oracle -- this is just for a sequence of hidden variables
    def oracle(self, hidden):
        for t1 in range(2,len(hidden)):
            if(hidden[t1] != hidden[t1-1]):
                for t2 in range(t1-1):
                    if(hidden[t1] == hidden[t2]):
                        return False
        return True

    
    def generate_hidden(self, *, t_max, num_observations = 1, seed=None, debug=False):
        H = [-1]*num_observations
        random.seed(seed)
        start_probs = self.start_probs
        trans_mat = self.trans_mat
        N = self.data.N
        
        #Deal with t_max
        #I want it to either be an integer of array
        # Check if t_max is an integer
        if isinstance(t_max, int):
            # Create a vector of the same length as num_observations filled with t_max
            t_max_vector = np.full(num_observations, t_max)
        elif isinstance(t_max, (list, np.ndarray)) and len(t_max) == num_observations:
            # If t_max is already a vector of the correct length, use it directly
            t_max_vector = np.array(t_max)
        else:
            raise ValueError("t_max must be an integer or a vector of the same length as num_observations.")
        
        for n in range(num_observations):
            H[n] = [-1]*t_max_vector[n]
            
            #Observation at time 0
            while True:
                probs = [start_probs[h] for h in range(N)]
                H[n][0] = np.random.choice(range(N), p=probs)
                
                for t in range(1,t_max):
                    probs = [trans_mat[H[n][t-1]][h] for h in range(N)]
                    H[n][t] = np.random.choice(range(N), p=probs)
                    
                if(self.oracle(H[n])):
                    break

        return np.array(H, dtype=int)

    #Needs to be slightly different than the previous generate_observations because we want
    #this to just depend upon a feasibly generated sequence of hidden states
    def generate_observations_from_hidden(self, H, *, return_obs = False):  
        emission_probs = self.emission_probs
        N = self.data.N
        O = [0]*len(H)
        
        for i in range(len(H)):
            O[i] = [0]*len(H[i])
            for t in range(len(H[i])):
                probs = [emission_probs[H[i][t]][o] for o in range(N)]
                O[i][t] = np.random.choice(range(N), p = probs)
        
        if(return_obs):
            return np.array(O, dtype=int)
        self.O = O 
                

    # Generate feasible hidden states based on observations
    # Does this using a rejection sampling approach
    # Observations are a variable in HMMBase as self.O
    def generate_from_observation(self, *, num = 1, seed=None):
        return -1
        


#
# MAIN
#
debug = False
seed = None

model = Citation()
model.load_process(filename="citation.yaml")
H = model.generate_hidden(t_max = 30, num_observations=1)
print(H)
print(model.generate_observations_from_hidden(H,return_obs=True))

print("FINISHED")