import argparse
import sys
xax = 8193
yax = 10
def mapping(s): # 0 is loss, 8193 is win, 1-8192 are intermediate states. this is a resultant one-one mapping function
    p1 = s[0:2]
    p2 = s[2:4]
    pr = s[4:6]
    p = s[6]
    p1_coor = int(p1)
    p2_coor = int(p2)
    pr_coor = int(pr)
    possession = int(p)
    mapped_value = 512*(p1_coor-1) + 32*(p2_coor-1) + 2*(pr_coor-1) + possession

    return mapped_value

def policy(path):
    file=open(path,'r')
    policy=file.read().split('\n')
    file.close()
    
    pi={}
    del policy[-1]
    
    for i in range(1,len(policy)):
        line=policy[i].split(" ")
        pi[line[0]]=(float(line[1]),float(line[2]),float(line[3]),float(line[4]))
    
    return pi

def change(state,dir,player):  #returns the state(intermediate)
    pos=int(state[player*2:player*2+2])
    if(dir==2 or dir==3):
        final_pos=-20 + 8*dir + pos
        if final_pos>0 and final_pos<=16:
            return state[:player*2]+str(int(final_pos/10))+str(int(final_pos%10))+state[player*2+2:]
        else:
            return "0101010"
    if(dir==0):
        final_pos=pos-1
        if pos%4!=1:
            return state[:player*2]+str(int(final_pos/10))+str(int(final_pos%10))+state[player*2+2:]
        else:
            return "0101010"
        
    if(dir==1):
        final_pos=pos+1
        if pos%4!=0:
            return state[:player*2]+str(int(final_pos/10))+str(int(final_pos%10))+state[player*2+2:]
        else:
            return "0101010"



def movement(state,prev_s,dir,player,p):
    
    if(player+2!=int(state[-1]) + 1):
        final_state=change(state,dir,player)
        prob=1-p
    else:
        final_state=change(state,dir,player)
        if(final_state[-1]=="0"):
            return final_state,1
        
        # Same square attained
        if(int(final_state[player*2:player*2+2])==int(state[4:6])):
            prob=0.5-p
         
        # Cross tackling
        elif(int(state[player*2:player*2+2])==int(state[4:6]) and int(final_state[player*2:player*2+2])==int(prev_s[4:6])):
            prob=0.5-p
        else:
            prob=1-2*p
       
    return final_state,prob    

def passing(state,q):
    pos1=int(state[0:2])
    pos2=int(state[2:4])
    posr=int(state[4:6])
    
    x1=(pos1-1)%4
    x2=(pos2-1)%4
    xr=(posr-1)%4
    y1=(pos1-1)//4
    y2=(pos2-1)//4
    yr=(posr-1)//4
    prob=q-0.1*max(abs(x1-x2),abs(y1-y2))  
    
    if((x1==xr and y1==yr) and (x2==xr and y2==yr)):       # same coordinates
        prob/=2
    elif(abs(x2-x1)==0 and  x1==xr and ((y1-yr)*(y2-yr))<=0):      # x1 and x2 lying on the same column
        prob/=2
    elif(abs(y1-y2)==0 and  y1==yr and ((x1-xr)*(x2-xr))<=0):      # y1 and y2 lying on the same row
        prob/=2
        
       
    elif( abs(x1-x2)!=0 and abs(y1-y2)!=0 and x1!=xr and x2!=xr ):                           
        if((y1-yr)/(x1-xr)==(yr-y2)/(xr-x2) and ((x1-xr)*(x2-xr))<0):
            prob/=2
    elif(abs(x1-x2)!=0 and abs(y1-y2)!=0):                                                     
        if(x1==xr and y1==yr and abs((y2-yr)/(x2-xr))==1): 
            prob/=2
        elif(x2==xr and y2==yr and abs((y1-yr)/(x1-xr))==1):
            prob/=2
       
    final_state=state[:-1]+str(3-int(state[-1]))
    return final_state,prob

def shooting(state,q):
    player=int(state[-1])-1
    pos=int(state[player*2:player*2+2])
    posr=int(state[4:6])
    x=(pos-1)%4
    
    prob=q-0.2*(3-x)
    
    if(posr==8 or posr==12):
        prob= prob/2
    return prob

def transitions(opp_pi,p,q):
    
    mdp=[[{} for a in range(yax)] for b in range(xax+1)]
    print("numStates 8194")
    print("numActions 10")
    print("end 0 8193")
    for s in opp_pi.keys():
        for a in range(10):
            mdp[mapping(s)][a][0]=0
        mdp[mapping(s)][9][xax]=0
        for i in range(4):
            if not (opp_pi[s][i]==0):
                intermediate_s=change(s,i,2) 
                for action in range(4):
                    final_state,prob=movement(intermediate_s,s,action,0,p)
                    
                    if(final_state[-1]=="0"):
                       mdp[mapping(s)][action][0]+=opp_pi[s][i] 
                    else:
                        mdp[mapping(s)][action][mapping(final_state)]=prob*opp_pi[s][i]
                        mdp[mapping(s)][action][0]+=(1-prob)*opp_pi[s][i]
                        
                for action in range(4,8):
                    final_state,prob=movement(intermediate_s,s,action-4,1,p)
                    
                    if(final_state[-1]=="0"):
                        mdp[mapping(s)][action][0]+=opp_pi[s][i]                       
                    else:
                        mdp[mapping(s)][action][mapping(final_state)]=prob*opp_pi[s][i]
                        mdp[mapping(s)][action][0]+=(1-prob)*opp_pi[s][i]
                final_state,prob=passing(intermediate_s,q)
                mdp[mapping(s)][8][mapping(final_state)]=prob*opp_pi[s][i]
                mdp[mapping(s)][8][0]+=(1-prob)*opp_pi[s][i]

                prob=shooting(intermediate_s,q)
                
                mdp[mapping(s)][9][xax]+=prob*opp_pi[s][i]
                mdp[mapping(s)][9][0]+=(1-prob)*opp_pi[s][i]
                
    for i in range(1,xax):
        for j in range(9):
            for k in mdp[i][j].keys():
                print("transition",i,j,k,0,mdp[i][j][k]) 
        for k in mdp[i][9].keys():
            if k==xax:
                print("transition",i,9,k,1,mdp[i][9][k])
            else:
                print("transition",i,9,k,0,mdp[i][9][k])
    print("mdptype episodic")
    print("discount  1")           

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opponent', type=str, required=True, help='policy path')
    parser.add_argument('--p', type=float, required=True, help='p')
    parser.add_argument('--q', type=float, required=True, help='q')
    args = parser.parse_args()
    po = policy(args.opponent)
    transitions(po,args.p,args.q)
