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
import pprint
from munch import Munch
import pyomo.environ as pe

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


    # Generate feasible hidden states based on observations
    # Does this using a rejection sampling approach
    # Observations are a variable in HMMBase as self.O
    def constrained_generate_from_observation(self, num = 1):
        return 1
        


#
# MAIN
#
debug = False
seed = None

model = Citation()
model.load_process(filename="citation.yaml")
print(model.constrained_generate_from_observation(2))

print("FINISHED")