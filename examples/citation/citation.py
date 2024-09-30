#TODO

import yaml
import random
import pyomo.environ as pe
import numpy as np
import sys

from chmmpy import HMMBase
from chmmpy.util import normalize_array


class Citation(HMMBase):
    def load_process(self, *, data=None, filename=None, seed=None):
        """
        Load HMM parameters from a YAML file or use provided data.

        Parameters:
        - data: Optional; a dictionary containing HMM parameters.
        - filename: Optional; path to a YAML file containing HMM parameters.

        Raises:
        - ValueError: If both data and filename are None.
        - KeyError: If required keys are missing in the data.

        Returns:
        - None
        """
        # Set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        
        #Loads from file
        if filename is not None:
            with open(filename, "r") as INPUT:
                data = yaml.safe_load(INPUT)
        assert data is not None

        alldata = self.data
        alldata.N = data["num_categories"]
        n = alldata.N #I got tired of writing alldata.H over and over again
        self.seed = seed
        self.name = data["name"]
        alldata.transition_prob = data["transition_prob"]
        alldata.noise_prob = data["noise_prob"]
        
        self.data.N = n
        
        # Initialize HMM parameters
        self.start_probs = [1 / n] * n
        self.emission_probs = [
            [
                (1 - alldata.noise_prob if i == j else alldata.noise_prob / (n - 1))
                for j in range(n)
            ]
            for i in range(n)
        ]
        self.trans_mat = [
            [
                (1 - alldata.transition_prob if i == j else alldata.transition_prob / (n - 1))
                for j in range(n)
            ]
            for i in range(n)
        ]

    def oracle(self, hidden):
        """
        Check if the sequence of hidden variables meets the oracle condition.

        The oracle condition is satisfied if:
        - The states appear in blocks that do not repeat

        Parameters:
        - hidden: A list or array of hidden states.

        Returns:
        - bool: True if the sequence meets the oracle condition, False otherwise.
        """
        seen_states = set()
        seen_states.add(hidden[0])
        seen_states.add(hidden[1])
    
        for t1 in range(2, len(hidden)):
            if hidden[t1] != hidden[t1 - 1]:
                # Check if the current state has appeared before
                if hidden[t1] in seen_states:
                    return False
                seen_states.add(hidden[t1])
        return True

    
    def generate_hidden(self, *, t_max, num_hidden = 1, debug=False):
        """
        Generate hidden states based on the given parameters.

        Parameters:
        - t_max: Either an integer or an array specifying the maximum time steps for each observation.
        - num_observations: Number of observations to generate.

        Returns:
        - A numpy array of hidden states.
        """
        start_probs = self.start_probs
        trans_mat = self.trans_mat
        N = self.data.N #Number of hidden variables
        
        #Deal with t_max
        # Check if t_max is an integer
        if isinstance(t_max, int):
            # Create a vector of the same length as num_hidden filled with t_max
            t_max_vector = np.full(num_hidden, t_max)
        elif isinstance(t_max, (list, np.ndarray)) and len(t_max) == num_hidden:
            # If t_max is already a vector of the correct length, use it directly
            t_max_vector = np.array(t_max)
        else:
            raise ValueError("t_max must be an integer or a vector of the same length as num_hidden.")
        
        hidden = [[0] * t_max_vector[i] for i in range(num_hidden)]
        
        for n in range(num_hidden):
            #Observation at time 0
            while True:
                probs = [start_probs[h] for h in range(N)]
                hidden[n][0] = np.random.choice(range(N), p=probs)
                
                for t in range(1,t_max):
                    probs = [trans_mat[hidden[n][t-1]][h] for h in range(N)]
                    hidden[n][t] = np.random.choice(range(N), p=probs)
                    
                if(self.oracle(hidden[n])): 
                    break

        return np.array(hidden, dtype=int)

    def generate_observations_from_hidden(self, hidden, *, return_obs = False): 
        """
        Generate observations from hidden states based on emission probabilities.

        Parameters:
        - H: A list of lists representing hidden states.
        - return_obs: A boolean indicating whether to return the observations. If false, observations are stored in HMM. 

        Returns:
        - An array of observations if return_obs is True; otherwise, None, but sets self.O
        """      
        emission_probs = self.emission_probs
        N = self.data.N #Number of possible observation variables
        observations = [[0] * len(hidden_state) for hidden_state in hidden]

        for i, hidden_state in enumerate(hidden):
            for t in range(len(hidden_state)):
                probs = [emission_probs[hidden_state[t]][o] for o in range(N)]
                observations[i][t] = np.random.choice(range(N), p=probs)

        if return_obs:
            return np.array(observations, dtype=int)
        
        self.O = observations
                
                
                
    def generate_hidden_from_observation(self, *, observations=None, num_hidden = 1):
        """
        Generates feasible sequences of hidden states based on the observations.

        Parameters:
        - observations: A list of observations. If not provided, self.O is used.
        - num_hidden: The number of hidden states for each observation we wish to generate.

        Returns:
        - An array of hidden variables (shape: (num_sequences, sequence_length)).
        """
        # Initialize model parameters
        start_probs = self.start_probs
        trans_mat = self.trans_mat
        emission_probs = self.emission_probs
        num_hidden_states = self.data.N  # Number of possible hidden variables

        # Use self.O if observations are not provided
        if observations is None:
            observations = self.O
            
        # Initialize the hidden states array
        # TODO: This doesn't work rn if the observations are of different lengths
        hidden_states = np.zeros((len(observations), num_hidden, max(len(obs) for obs in observations)), dtype=int)

        for i, single_observation in enumerate(observations):
            for j in range(num_hidden):
                while True:
                    # Calculate initial probabilities for the first hidden state
                    initial_probs = [start_probs[h] * emission_probs[h][single_observation[0]] for h in range(num_hidden_states)]
                    initial_probs = normalize_array(initial_probs)
                    hidden_states[i][j][0] = np.random.choice(range(num_hidden_states), p=initial_probs)

                    # Generate the remaining hidden states
                    for t in range(1, len(single_observation)):
                        transition_probs = [trans_mat[hidden_states[i][j][t - 1]][h] * emission_probs[h][single_observation[t]] for h in range(num_hidden_states)]
                        transition_probs = normalize_array(transition_probs)
                        hidden_states[i][j][t] = np.random.choice(range(num_hidden_states), p=transition_probs)

                    # Check if the generated sequence is valid
                    if self.oracle(hidden_states[i][j]):
                        break

        return np.array(hidden_states, dtype= int)

    
    def perturb_parameters(self, perturb_val):
        """
        Perturbs the model parameters: start_probs, trans_mat, and emission_probs
        multiplicatively by perturb_val^c, where c is uniformly distributed between -1 and 1.
        All parameters are renormalized after perturbation.

        Parameters:
        - perturb_val: A value >= 1 where a larger value means more perturbation.
        
        Returns:
        - None
        """

        # Perturb and normalize a 1D array
        def perturb_and_normalize(array: np.ndarray) -> np.ndarray:
            perturbed_array = np.copy(array)
            for i in range(len(perturbed_array)):
                c = random.uniform(-1, 1)
                perturbed_array[i] *= perturb_val ** c
            return normalize_array(perturbed_array)

        # Perturb and normalize start probabilities
        self.start_probs = perturb_and_normalize(self.start_probs)

        # Perturb and normalize transition matrix
        self.trans_mat = np.array([perturb_and_normalize(row) for row in self.trans_mat])

        # Perturb and normalize emission probabilities
        self.emission_probs = np.array([perturb_and_normalize(row) for row in self.emission_probs])
            
    def learn(self, observations, *, num_hidden=1, learn_param=-2/3, convergence_factor = 1E-6, num_it = 0, noisy = True):
        """
        Learns the model parameters: start_probs, trans_mat, and emission_probs
        based on the observations.
        Uses a rejection based sampling approach.

        Parameters:
        - observations: Observations we do learning with 
        - num_hidden: The number of random states that are generated each step in the learning process
        - learn_param: A value in [-1,-1/2) more negative values mean that previously generated hidden states are weighted more heavily
        - convergence_factor: If the parameters change less than this in an iteration, the algorithm terminates
        - num_it: The starting value of number of iterations, a larger value implies that you trust the initial guess of parameters more
        - noisy: Determines whether the amount of change in the model parameters is output every time
        
        Returns:
        - None
        """   
        # Initialize model parameters
        # We use star to denote the fact that the parameters are not normalized
        start_probs_star = self.start_probs
        trans_mat_star = self.trans_mat
        emission_probs_star = self.emission_probs
        
        #Update the parameters
        while True:
            num_it += 1
            alpha = pow(num_it,learn_param)
            
            #Reweight previous matrices
            start_probs_star = start_probs_star*(1-alpha)
            emission_probs_star = emission_probs_star*(1-alpha)
            trans_mat_star = trans_mat_star*(1-alpha)
            
            H = self.generate_hidden_from_observation(observations=observations, num_hidden=num_hidden)
            #np.set_printoptions(threshold=sys.maxsize)
            #print(observations)
            #print(H)
            #break
            
            for r, mat in enumerate(H):
                for n, vec in enumerate(mat):
                    start_probs_star[vec[0]] += alpha
                    
                    for t in range(1,len(vec)):
                        emission_probs_star[vec[t]][observations[r][t]] += alpha
                        trans_mat_star[vec[t-1]][vec[t]] += alpha
                
            #Normalize and compare to the parameters in the HMM        
            start_probs = normalize_array(start_probs_star)
            emission_probs = np.array([normalize_array(row) for row in emission_probs_star])
            trans_mat = np.array([normalize_array(row) for row in trans_mat_star])
            
            # Calculate the absolute differences
            differences_start_probs = np.abs(start_probs-self.start_probs)
            differences_emission_probs = np.abs(emission_probs-self.emission_probs)
            differences_trans_mat = np.abs(trans_mat-self.trans_mat)
            val = max(np.max(differences_start_probs), np.max(differences_emission_probs), np.max(differences_trans_mat))
            
            #Update HMM parameters
            self.start_probs = start_probs
            self.emission_probs  = emission_probs
            self.trans_mat = trans_mat
            
            if noisy:
                print(val)
            
            if val < convergence_factor:
                break

#
# MAIN
#
'''
debug = False
seed = None

model = Citation()
model.load_process(filename="citation.yaml")
H = model.generate_hidden(num_hidden = 500, t_max=25)
O = model.generate_observations_from_hidden(H, return_obs=True)

model.perturb_parameters(2)
print("Before learning:")
print(model.start_probs)
print(model.emission_probs)
print(model.trans_mat)

model.learn(O, learn_param=-1, convergence_factor=1E-3, num_it=0, num_hidden=1)

print("After learning:")
print(model.start_probs)
print(model.emission_probs)
print(model.trans_mat)


print("FINISHED")
'''