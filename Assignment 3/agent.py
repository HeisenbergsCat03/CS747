import os
import sys
import random 
import json
import math
import utils
import time
import numpy as np
import config
import numpy
random.seed(73)

class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.ns = utils.NextState()


    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius
    
    def force_mapping(self,d,f_max = 1): #should decrease as d decreases
        return (6.15/7)*f_max/(1.075+math.exp(-d/600))
    
    def check(self, target_ball,white):
        pockets = []
        angles = []
        for i in range(6):
            hole = np.array(self.holes[i])
            v1 = (hole)-(target_ball)
            v2 = (target_ball) - (white)
            if 2*self.ball_radius/np.linalg.norm(v2)>1:
                para = 1
            else:
                para = 2*self.ball_radius/np.linalg.norm(v2)
            if self.angle_bw_2vect(v1,v2)<math.acos(para)*180/math.pi:
                pockets.append(i)
                angles.append(self.angle_bw_2vect(v1,v2))
        if len(pockets)==0:
            return None
        else:
            d0 = np.linalg.norm(np.array(self.holes[pockets[0]])-np.array(target_ball))
            idx = 0
            idx2 = 0
            angle0 = angles[0]
            for i in range(len(pockets)):
                if angles[i]<angle0:
                    angle0 = angles[i]
                    idx2 = i
                dist = np.linalg.norm(np.array(self.holes[pockets[i]])-np.array(target_ball))
                if dist<d0:
                    d0 = dist
                    idx = i
            

            return pockets[idx2]
    def closest_ball(self,ball_pos):
        min_dist = float('inf')
        closest_ball = None
        for i in ((ball_pos)):
            if i == 0 or i == 'white':
                continue
            dist = np.linalg.norm(np.array(ball_pos[i])-np.array(ball_pos[0]))
            if dist < min_dist:
                min_dist = dist
                closest_ball = i
        return closest_ball
    
    def final_angle(self,angle):
        deg = angle*180/math.pi
        if deg>0 and deg<90:
            return -((90+deg))*math.pi/180 
        elif deg>=90 and deg<=180:
            return ((270-deg))*math.pi/180 
        elif deg<=0 and deg>-90:
            return -(90+deg)*math.pi/180 
        elif deg>=-180 and deg<=-90:
            return -((90+deg))*math.pi/180 
        
    def closest_hole(self,ball_pos, ball_idx):
        min_dist = float('inf')
        closest_hole = None
        for i in range(len(self.holes)):
            dist = np.linalg.norm(np.array(self.holes[i])-np.array(ball_pos[ball_idx]))
            if dist < min_dist:
                min_dist = dist
                closest_hole = i
        return closest_hole
    
    def angle_bw_2vect(self,v1,v2):
        return math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))*180/math.pi
    
    def action(self, ball_pos=None):
        white = np.array(ball_pos[0])
        best_score = float('-inf')
        best_action = None
        
        white = np.array(ball_pos[0])
        target_ball = np.array(ball_pos[self.closest_ball(ball_pos)])
        if self.check(target_ball,white) is None:
            target_pocket = np.array(self.holes[self.closest_hole(ball_pos,self.closest_ball(ball_pos))])
        else:
            target_pocket = np.array(self.holes[self.check(target_ball,white)])
        
        #print(target_pocket)
        ys = target_ball[1]
        xs = target_ball[0]
        xc = white[0]
        yc = white[1]
        xp = target_pocket[0]
        yp = target_pocket[1]
        theta_p = math.atan2((yp-ys),(xp-xs))
        theta_v = math.atan2((ys-yc- 2*self.ball_radius*math.sin(theta_p)),(xs-xc- 2*self.ball_radius*math.cos(theta_p)))
        dist = np.linalg.norm(np.array(ball_pos[0])-target_ball) + np.linalg.norm(target_ball-target_pocket)
        best_action = (self.final_angle(theta_v)/math.pi, self.force_mapping(dist))
        #print(best_action)
        return best_action
