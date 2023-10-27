import random, sys, time
import argparse
parser = argparse.ArgumentParser()
import numpy as np
import pulp
class MDP():
    def __init__(self, path, algorithm, policy):
        self.mdp = {}
        self.path = path
        self.algorithm = algorithm
        self.policy = policy
        self.readMDP()
        self.v_star = np.zeros(self.mdp['numStates'])
        self.pi_star = np.zeros(self.mdp['numStates'],dtype=int)
        self.policyfile = policy
        if(self.policyfile == '-1'):
            if(self.algorithm == 'lp'):
                self.lp()
            elif(self.algorithm == 'vi'):
                self.vi()
            elif(self.algorithm == 'hpi'):
                self.hpi()
            self.print_output() 
        else:
            self.readPolicy()
            self.print_output()
        
    def readMDP(self):
        with open(self.path) as file:
            for line in file:
                w = line.split()
                if w[0] == 'numStates':
                    self.mdp['numStates'] = int(w[-1])
                elif w[0] == 'discount':
                    self.mdp['gamma'] = eval(w[-1])
                elif w[0] == 'numActions':
                    self.mdp['numActions'] = int(w[-1])
                    states = self.mdp['numStates']
                    actions = self.mdp['numActions']
                    self.mdp['transition'] = np.zeros((states, actions, states))
                    self.mdp['reward'] = np.zeros((states, actions, states))
                elif w[0] == 'end':
                    l = w[1:]
                    self.mdp['end'] = list(map(int, l))
                elif w[0] == 'transition':
                    l = w[1:]
                    l = list(map(eval, l))
                    s, a, s_, r, p = l
                    self.mdp['transition'][s][a][s_] = p
                    self.mdp['reward'][s][a][s_] = r
                elif w[0] == 'mdptype':
                    self.mdp['mdptype'] = w[-1]

    def readPolicy(self):
        c = 0
        with open(self.policyfile) as file:
            while(line := file.readline()):
                self.pi_star[c] = int(line)
                c += 1
        tr_pi = self.mdp['transition'][np.arange(self.mdp['numStates']),self.pi_star]
        r_pi = self.mdp['reward'][np.arange(self.mdp['numStates']),self.pi_star]
        rpi = np.sum(tr_pi*r_pi,axis=1,keepdims=True)
        #print(tr_pi.shape)
        v =  np.linalg.inv(np.identity(self.mdp['numStates']) - self.mdp['gamma']*tr_pi)@rpi
        #print(self.v_star.shape)
        self.v_star = np.squeeze(v)


    def value_i(self,V,policy):
        rew = (self.mdp['reward'] + self.mdp['gamma']*V)
        x = self.mdp['transition']*rew
        x_ = np.sum(x,axis = -1)
        if(policy):
            return np.argmax(x_,axis = -1)
        else:
            return np.max(x_,axis = -1)


    def vi(self):
        t = np.random.randn(self.mdp['numStates'])
        while True:
            self.v_star = self.value_i(t,False) 
            if(np.linalg.norm(self.v_star - t) < 1e-10):
                break   
            t = self.v_star
        self.pi_star = self.value_i(self.v_star,True)


    def lp(self):
        v = pulp.LpVariable.dicts("s",range(self.mdp['numStates']))
        prob = pulp.LpProblem("MDP",pulp.LpMinimize)
        prob += sum(v[i] for i in range(self.mdp['numStates']))
        for i in range(self.mdp['numStates']):
            for j in range(self.mdp['numActions']):
                prob += v[i] >= self.mdp['gamma']*sum(self.mdp['transition'][i][j][k]*v[k] for k in range(self.mdp['numStates'])) + sum(self.mdp['transition'][i, j, k]*self.mdp['reward'][i, j, k] for k in range(self.mdp['numStates']))
        prob.solve(pulp.apis.PULP_CBC_CMD(msg = 0))

        for i in range(self.mdp['numStates']):
            self.v_star[i] = v[i].varValue
        self.pi_star = self.value_i(self.v_star,True)
        
    def hpi(self):
        pi = np.random.randint(low = 0, high = self.mdp["numActions"],size = self.mdp["numStates"])
        temp = pi
        while (True):
            transit = self.mdp['transition'][np.arange(self.mdp["numStates"]), temp]
            rew = self.mdp["reward"][np.arange(self.mdp["numStates"]), temp]
            
            rpi = np.sum(transit*rew,axis=1,keepdims=True)
            self.v_star= np.linalg.inv(np.eye(self.mdp["numStates"]) - self.mdp["gamma"] * transit)@rpi
            self.v_star = np.squeeze(self.v_star)

            self.pi_star = np.argmax(np.sum(self.mdp["transition"] * (self.mdp["reward"] + self.mdp["gamma"] * self.v_star), axis=-1), axis=-1)
            if(np.all(temp == self.pi_star)):
                break
            temp = self.pi_star

    def print_output(self):
        for i in range(self.mdp['numStates']):
            print("{:.6f}".format(self.v_star[i]) + ' ' + str(self.pi_star[i]))

    def solveMDP(self):
        if(self.algorithm == "vi"):
            self.vi()
        elif(self.algorithm == "hpi"):
            self.hpi()
        elif(self.algorithm == "lp"):
            self.lp()


parser.add_argument("--mdp",type=str,help='File containing the MDP description', required=True)
parser.add_argument("--algorithm",type=str,default='vi')
parser.add_argument("--policy",type=str,default='-1')

args = parser.parse_args()
MDP(args.mdp,args.algorithm,args.policy)