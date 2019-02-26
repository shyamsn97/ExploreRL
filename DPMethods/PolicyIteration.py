import numpy as np
from .Agent import Agent
class PolicyIteration(Agent):
    """
        Policy Iteration: 
            Reference: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf
            Parameters:
                value_map : state values for each state
                policy_map : greedy action choice for each state
                env : gym environment
    """
    def __init__(self,env):
        super().__init__(env)
        self.values = np.zeros(env.nS)
        self.policy = np.array([np.random.choice(env.nA) for i in range(env.nS)])
        self.eps = 1e-10
        self.gamma = 1
        self.policy_iteration()
        
    def act(self,observation):
        return self.policy[observation]
        
    def evaluate_policy(self):
        converged = False
        while True:
            new_values = np.zeros(self.env.nS)
            for s in range(self.env.nS):
                p_a = self.policy[s]
                for p , next_s, r, done in self.env.P[s][p_a]:
                    new_values[s] += p*(r + self.gamma*self.values[next_s]) 
            if np.sum(np.abs(new_values - self.values)) <= self.eps:
                break
            for i in range(self.env.nS):
                self.values[i] = new_values[i]
    
    def improve_policy(self):
        for s in range(self.env.nS):
            q_vals = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for p , next_s, r, done in self.env.P[s][a]:
                    q_vals[a] += p*(r + self.gamma*self.values[next_s])
            self.policy[s] = np.argmax(q_vals)
    
    def policy_iteration(self,max_iterations=100):
        for i in range(max_iterations):
            old_values = self.values.copy()
            self.evaluate_policy()
            self.improve_policy()
            if np.sum(np.abs(old_values - self.values)) <= self.eps:
                break