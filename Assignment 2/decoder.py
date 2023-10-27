import numpy as np
import sys, random, time, os
import matplotlib.pyplot as plt
import argparse
#sys.stdout = open('output5.txt', 'w',encoding='utf-8')



def opponent_key(path):
    f=open(path,'r')
    policy=f.read().split('\n')
    f.close()
    del policy[-1]
    
    states=[]
    for i in range(1,len(policy)):
        line=policy[i].split(" ")
        states.append(line[0])
    
    return states
def value_file(path_val):
    file = open(path_val,'r')
    value = file.read().split("\n")
    file.close()
    v = []
    pi = []
    del value[-1]
    for i in range(1,len(value)):
    #   line = value[i].split(" ")
          v_=value[i].split(" ")[0]
          pi_=value[i].split(" ")[-1]
    #     v_ = line[0]
    #     pi_ = line[-1]
          v.append(float(v_))
          pi.append(int(pi_))
    return v,pi
        


def print_output(states,v,pi):
    for i in range(len(states)):
        print(states[i] + ' ' + str(pi[i]) + ' ' + str(v[i]))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', help='path to value file', required=True)
    parser.add_argument('--opponent', help='path to opponents file', required=True)
    args = parser.parse_args()
    path_opp = args.opponent
    path_val = args.value_policy
    states = opponent_key(path_opp)
    v, pi = value_file(path_val)
    print_output(states,v,pi)

