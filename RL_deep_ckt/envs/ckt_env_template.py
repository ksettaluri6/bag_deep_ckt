"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces

import numpy as np
import random
import scipy.interpolate as interp
import scipy.optimize as sciopt

from collections import OrderedDict
import yaml
import yaml.constructor
import statistics
import os
import IPython
import importlib
import itertools

from bag_deep_ckt.util import *

#helper functions
def merge_keys(list_of_kwrds):
        merged_list = "/".join(list_of_kwrds)
        return merged_list

def break_hierarchy_into_lists(main_dict):
    kwrds, values = [], []
    for k, v in main_dict.items():
        if isinstance(v, dict):
            # internet check if v is not empty is equivalent to not v
            if v:
                ret_kwrds, ret_values = break_hierarchy_into_lists(v)
                for ret_kwrd in ret_kwrds:
                    updated_key = merge_keys([k, ret_kwrd])
                    kwrds.append(updated_key)
                values += ret_values
            else:
                # especial base case
                kwrds += [k]
                values += [None]
        else:
            # base case
            kwrds += [k]
            values += [v]
    return kwrds, values

def break_hierarchy(main_dict):
    kwrds, values = break_hierarchy_into_lists(main_dict)
    # remove None value entries
    ret_dict = {}
    
    for k, v in zip(kwrds, values):
      if v == None:
        continue
      ret_dict[k] = v
    return ret_dict

class CktName(gym.Env):
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    def __init__(self, multi_goal=False):
        design_specs_fname = "path/to/yaml/file"
        with open(design_specs_fname, 'r') as f:
            content = yaml.load(f)

        #initialize BAG environment stuffs
        eval_module = importlib.import_module(content['eval_core_package'])
        eval_cls = getattr(eval_module, content['eval_core_class'])
        self.eval_core = eval_cls(design_specs_fname=design_specs_fname)

        self.multi_goal = multi_goal

        # design specs
        self.specs = content['spec_range']

        # param array
        self.params = break_hierarchy(content['params'])
        self.params_vec = {}
        self.search_space_size = 1

        #extract all the params in seg_dict and iterate through them
        for key, value in self.params.items():
             if value is not None:
               # self.params_vec contains keys of the main parameters and the corresponding search vector for each
               self.params_vec[key] = np.arange(value[0], value[1], value[2]).tolist()
               self.search_space_size = self.search_space_size * len(self.params_vec[key])
        self.num_params = len(self.params_vec.values())
        
        # define action space
        self.action_meaning = [-1,0,5]
        self.action_space = spaces.Discrete(len(self.action_meaning)**self.num_params)
        self.observation_space = spaces.Box(
            low=np.append(np.array([CktName.PERF_LOW]*2*len(self.specs)),-5.0*np.ones(self.num_params)),
            high=np.append(np.array([CktName.PERF_HIGH]*2*len(self.specs)),5.0*np.ones(self.num_params)))

        #define action space, add the param ranges here 
        self.action_arr = list(itertools.product(*([self.action_meaning for i in range(len(self.params_vec.values()))])))

        #define reset at mid-point of length of each param array
        self.cur_params_idx = []
        for param_arr in self.params_vec.values():
            self.cur_params_idx.append(int(len(param_arr)/2))

        self.id_encoder = IDEncoder(self.params_vec)
        self.global_g = np.array([spec[0] for spec in list(self.specs.values())])
    
    def reset(self):

        #multi-goal feature
        if self.multi_goal == False:
            self.specs_ideal = self.global_g 
        else:
            rand_oidx = random.randint(0,self.num_os-1)
            self.specs_ideal = self.specs_list[rand_oidx]
            self.specs_ideal = []
            for spec in list(self.specs.values()):
               self.specs_ideal.append(spec[rand_oidx])
            self.specs_ideal = np.array(self.specs_ideal)

        #format arr into Design class
        design = Design(self.specs, self.id_encoder, list(self.cur_params_idx))
        design_results = self.eval_core.evaluate([design]) 

        #parse simulation results
        self.cur_spec = self.parse_output(design_results)
       
        #normalize goal spec and current spec to some constant global_g
        self.ideal_spec_norm = self.lookup(np.array(self.specs_ideal), self.global_g)
        cur_spec_norm = self.lookup(np.array(self.cur_spec), self.global_g)
        
        #reward function is just the negative cost
        reward = -1*design_results[0]['cost']

        #formulate observation arr
        self.ob = np.concatenate([cur_spec_norm, self.ideal_spec_norm, self.cur_params_idx])
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """
        self.cur_params_idx = self.cur_params_idx + np.array(self.action_arr[int(action)])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0 for param in range(len(list(self.params.values())))], [(len(param_vec)-1) for param_vec in list(self.params_vec.values())])

        #format arr into Design class and run simulator
        design = Design(self.specs, self.id_encoder, list(self.cur_params_idx))
        design_results = self.eval_core.evaluate([design]) 

        #parse sim results
        self.cur_spec = self.parse_output(design_results)
        cur_spec_norm = self.lookup(np.array(self.cur_spec), self.global_g)

        #negative cost
        reward = -1*design_results[0]['cost']
        done = False

        #incentivize reaching goal state
        if (reward >= 10):
            done = True
            print('-'*10)
            print('params = ', self.cur_params_idx)
            print('specs:', cur_specs)
            print('ideal specs:', self.specs_ideal)
            print('re:', reward)
            print('-'*10)

        self.ob = np.concatenate([cur_spec_norm, self.ideal_spec_norm, self.cur_params_idx])
        return self.ob, reward, done, None 

    def parse_output(self, design_results):
        cur_spec = []
        #extract relevant specs from design class
        for spec in design_results[0].keys():
          if (spec == 'valid') or (spec == 'cost'):
            continue
          else:
            cur_spec.append(design_results[0][spec])
        return cur_spec

    def lookup(self, spec, goal_spec):
        norm_spec = (spec-goal_spec)/goal_spec
        return norm_spec
    
def main():
  env = CktName()
  env.reset()
  env.step(#here)

if __name__ == '__main__':
  main()
