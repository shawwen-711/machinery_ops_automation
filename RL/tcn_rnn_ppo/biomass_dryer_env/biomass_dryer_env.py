#
#  Copyright 2025 Battelle Energy Alliance, LLC.  All rights reserved.
# 

import time, os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np
import pandas as pd
import torch
# torch.use_deterministic_algorithms(True)
import torch.nn as nn
from collections import deque

from tcn_models.ntrgt_tcn_model import TCNMultiTarget



class BiomassDryerEnv(gym.Env):
    """
    A custom Gym environment for dryer control that:
      - Uses multiple TCNMultiTarget models (one for each target group) to predict the next state 
        from a rolling buffer.
      - In offline mode, the buffer is initialized from training data (setup_info['data']) using the feature order in all_vars.
      - In real-time mode, the environment only sends control commands via step()'s info={'ctrl_signal':...}, waits for time_step_sec,
        and then reads back the state via set_real_time_state() taking args from outside of the class instance
      
      The setup_info dict should include:
         - data: a DataFrame with historical data.
         - seq_len: length of the rolling window.
         - ctrl_vars: list of control variable names.
         - state_vars: list of state variable names.
         - trgt_group_names: list of target group names (order matters).
         - trgt_vars: dict mapping each group name to its list of variable names.
         - ftr_scalers: dict mapping each group name to its feature scaler.
         - trgt_scalers: dict mapping each group name to its target scaler.
         - state_dict_paths: dict mapping each group name to its model state_dict path.
    """
    def __init__(self, setup_info, trgt_output_mc=35.0, start_idx=0, real_time_mode=False, 
                 time_step_sec=2.0, max_timesteps=float('inf'), device='cpu'):
        
        super().__init__()
        
        self.real_time_mode = real_time_mode
        self.time_step_sec = time_step_sec
        self.seq_len = setup_info['seq_len']
        self.trgt_output_mc = trgt_output_mc
        self.start_idx = start_idx
        self.buffer = deque(maxlen=self.seq_len)
        self.real_time_state= None
        
        # Setup info extraction.
        self.data = setup_info['data']
        self.ctrl_vars = setup_info['ctrl_vars']
        self.state_vars = setup_info['state_vars']
        self.all_vars = self.ctrl_vars + self.state_vars
        
        self.trgt_group_names = setup_info['trgt_group_names']
        self.trgt_vars = setup_info['trgt_vars']
        self.ftr_scalers = setup_info['ftr_scalers']
        self.trgt_scalers = setup_info['trgt_scalers']
        self.state_dict_paths = setup_info['state_dict_paths']

        # Used to search for the closest data point in the historical data
        self.euc_state_vars = [var for group in self.trgt_group_names 
                               if group not in ['wip_p2', 'wip_p3']
                               for var in setup_info['trgt_vars'][group]]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else device)
        
        # Load TCN models for the groups that use them (i.e. all except wip_p2 and wip_p3)
        self._load_tcn_models()
        
        self.max_timesteps = max_timesteps
        self._init_action_space()
        self._init_state_space()
            
        self.crnt_timestep = 0
        self.reset()
    
    def _init_action_space(self):
        # Use the control variables from the training DataFrame.
        ctrl_df = self.data[self.ctrl_vars]
        ctrl_bounds = np.array(ctrl_df.apply(lambda x: (np.round(min(x)).astype(int),
                                                        np.round(max(x)).astype(int))))
        self.ctrl_dim = ctrl_bounds.shape[1]  # number of control variables
        self.action_low = np.array(ctrl_bounds[0], dtype=np.float32)
        self.action_high = np.array(ctrl_bounds[1], dtype=np.float32)
        
        # For the bed feed rate (assumed to be at index 1), use its unique values for discretization.
        self.bed_feed_rate_options = ctrl_df[self.ctrl_vars[1]].unique().astype(np.float32)
        
        self.action_dim = self.ctrl_dim + 1  # additional update flag
        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
            shape=(self.action_dim,),
            dtype=np.float32
        )
    
    def _init_state_space(self):
        # Use the state_vars from the training DataFrame.
        state_df = self.data[self.state_vars]
        state_bounds = np.array(state_df.apply(lambda x: (np.floor(min(x)), np.ceil(max(x)))))
        self.state_dim = state_bounds.shape[1]
        self.observation_space = spaces.Box(
            low=np.array(state_bounds[0], dtype=np.float32),
            high=np.array(state_bounds[1], dtype=np.float32),
            shape=(self.state_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        self.in_target_zone = False
        self._np_random, seed = seeding.np_random(seed)
        
        if self.crnt_timestep > 0:
            print(f"Reset after {self.crnt_timestep} steps.")
            self.start_idx = self._np_random.integers(0, self.data.shape[0]-self.seq_len)
        else:
            self.start_idx = self._np_random.integers(0, self.data.shape[0]-self.seq_len) if self.start_idx is None else self.start_idx            
        
        self.crnt_timestep = 0
        
        # Safety check to ensure start_idx + seq_len is within bounds.
        if self.start_idx + self.seq_len > self.data.shape[0]:
            self.start_idx = self.data.shape[0] - self.seq_len
        print(f"start index: {self.start_idx}")

        self.prev_ctrl = np.zeros(self.ctrl_dim, dtype=np.float32)
        
        if self.real_time_mode:
            self.real_time_state = np.zeros(self.state_dim, dtype=np.float32) if self.real_time_state is None else self.real_time_state
            self.crnt_state = self.real_time_state
            
        else:
            # Initialize current state from training data (using the state_vars columns).
            self.crnt_state = self.data.loc[self.start_idx + self.seq_len - 1, 
                                            self.state_vars].to_numpy(dtype=np.float32)
            
            # Offline mode: clear and fill the buffer using the order of all_vars.
            self.buffer.clear()
            for i in range(self.seq_len):
                idx = self.start_idx + i
                row = self.data.loc[idx, self.all_vars].to_numpy(dtype=np.float32)
                self.buffer.append(row)
       
        return self.crnt_state, {}
    
    def _discretize_bed_feed_rate(self, value):
        options = self.bed_feed_rate_options
        idx = np.abs(options - value).argmin()
        return options[idx]
    
    def step(self, action):
        # Clip the normalized action vector.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        norm_ctrl = action[:self.ctrl_dim]
        update_flag = action[self.ctrl_dim]
        
        # Scale the normalized control action.
        scaled_ctrl = self.action_low + norm_ctrl * (self.action_high - self.action_low)
        scaled_ctrl[1] = self._discretize_bed_feed_rate(scaled_ctrl[1])
        
        # Update control based on update flag.
        if update_flag > 0.5:
            self.prev_ctrl = scaled_ctrl.copy()
            crnt_ctrl = self.prev_ctrl
        else:
            crnt_ctrl = self.prev_ctrl
        
        if self.real_time_mode:
            '''In real-time mode, read real-time-states from sensor data'''
            next_state = self.real_time_state  # return self.real_time_state
        
        else:
            # Offline mode: use the rolling buffer and predict the next state.
            predicted_state = self._predict_next_state(crnt_ctrl)
            # Compute penalty for any predicted state out of bounds.
            out_of_bound_penalty = self._compute_out_of_bound_penalty(predicted_state)
            # Find the historical particle size distribution states that's closest to the predicted state.
            next_state = self._aug_nearest_hist_psd(predicted_state)
        
        self.crnt_state = next_state
        
        ### Out-of-bound & Drifting Penalties & Rewards ###
        reward = self._compute_rewards(crnt_ctrl)
        # In real-time mode out_of_bound_penalty might be zero (or computed elsewhere).
        if not self.real_time_mode:
            reward -= out_of_bound_penalty
        
        abs_diff = abs(self.crnt_state[0] - self.trgt_output_mc)
        
        in_zone_now = abs_diff <= 2.0
        if in_zone_now:
            # Give a high bonus if the agent just entered the target zone,
            # and a smaller bonus if it remains there.
            if not self.in_target_zone:
                reward += 300  # bonus for entering the target zone
            else:
                reward += 150  # bonus for staying in the target zone
        else:
            # If the agent was previously in the target zone but has drifted out,
            # penalize it more steeply based on how far outside the zone it is.
            if self.in_target_zone:
                # Calculate penalty only for the overshoot beyond 2%
                drift_penalty = 50 * (abs_diff - 2.0)
                reward -= drift_penalty
        
        self.in_target_zone = in_zone_now       
        ### ----------------------------------------- ###
        
        self.crnt_timestep += 1
        terminated = self.crnt_timestep >= self.max_timesteps
        truncated = False
        
        ### ----------------------------------------- ###
        crnt_ctrl_dict = {"_master_cont_mc_output": crnt_ctrl[0] ,
                          "_dryer_feed_1": crnt_ctrl[1], 
                          "_supply_feed_2": crnt_ctrl[2]}
        
        info = {"update_flag": update_flag, 
                "ctrl_sigs": crnt_ctrl_dict}
        
        return self.crnt_state, reward, terminated, truncated, info
    
    def _compute_out_of_bound_penalty(self, pred_next_state):
        """
        Compute penalty based only on the deviation of the predicted state for the state variables
        belonging to the first five target groups (stored in self.euc_state_vars).
        """
        penalty = 0.0
        # Find indices in the full state vector corresponding to the first 5 target groups.
        indices = [self.state_vars.index(var) for var in self.euc_state_vars]
        
        for i in indices:
            if pred_next_state[i] < self.observation_space.low[i]:
                penalty += self.observation_space.low[i] - pred_next_state[i]
            elif pred_next_state[i] > self.observation_space.high[i]:
                penalty += pred_next_state[i] - self.observation_space.high[i]
        
        penalty_multiplier = 2
        return penalty * penalty_multiplier
    
    def _compute_rewards(self, crnt_ctrl):
        """
        Compute the reward based on the current output moisture (crnt_mc) and control action.
        The reward system is designed to:
          - Give a high bonus if the moisture content is very close to the target (35%).
          - Apply a base penalty proportional to the squared deviation from 35%.
          - Apply a severe penalty for any moisture content above 38%.
        """
        crnt_mc = self.crnt_state[0]
        action_penalty = np.sum(np.square(crnt_ctrl)) * 0.01
        # Difference from target (35%)
        diff = crnt_mc - self.trgt_output_mc
        abs_diff = abs(diff)
        
        # Bonus rewards for being close to 35%
        if abs_diff <= 1.0:
            reward_bonus = 300
        elif abs_diff <= 2.0:
            reward_bonus = 150
        elif abs_diff <= 3.0:
            reward_bonus = 50
        else:
            reward_bonus = 0
        
        # Base penalty: quadratic penalty based on deviation from 35%
        base_penalty = diff ** 2
        
        # Additional severe penalty for any moisture above 38%
        extra_penalty = 0
        if crnt_mc > 36.0:
            extra_penalty = (crnt_mc - 36.0) ** 2 * 50  # Increased multiplier for severe penalty
        elif crnt_mc < 30.0:
            extra_penalty = (30.0 - crnt_mc) ** 2 * 20  # Increased multiplier for severe penalty            
    
        reward = -base_penalty - action_penalty - extra_penalty + reward_bonus
        return reward

    def render(self, mode="human"):
        print("Current state:", self.crnt_state)
    
    def _load_tcn_models(self):
        """
        Instantiate and load the TCNMultiTarget models for each target group,
        except for groups 'wip_p2' and 'wip_p3'.
        The models are stored in the dictionary self.tcn_models, keyed by target group name.
        """
        self.tcn_models = {}
        for group in self.trgt_group_names:
            if group not in ['wip_p2', 'wip_p3']:
                model = TCNMultiTarget(self.all_vars, self.trgt_vars[group])
                state_dict_path = self.state_dict_paths[group]
                model.load_state_dict(torch.load(state_dict_path, weights_only=True, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.tcn_models[group] = model
                
    def _standard_scale_features(self, x, scaler=None):
        """
        Scales features using Standard (z-score) scaling.
        x: numpy array of shape (samples, num_features, seq_length)
        Returns: scaled x and the fitted scaler.
        """
        samples, num_features, seq_length = x.shape
        # Reshape to (samples*seq_length, num_features)
        x_reshaped = x.transpose(0, 2, 1).reshape(-1, num_features)
        if scaler is None:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_reshaped)
        else:
            x_scaled = scaler.transform(x_reshaped)
        # Reshape back to (samples, seq_length, num_features) then transpose to (samples, num_features, seq_length)
        x_scaled = x_scaled.reshape(samples, seq_length, num_features).transpose(0, 2, 1)
        return x_scaled, scaler
        
    def _predict_next_state(self, crnt_ctrl):
        """
        Update the rolling buffer with the current control and state,
        then use the TCN models that have been instantiated to predict their outputs.
        Returns:
            np.array: A concatenated prediction vector from the groups with loaded TCN models.
        """
        row = np.concatenate([crnt_ctrl, self.crnt_state]).astype(np.float32)
        self.buffer.append(row)
        seq_data = np.array(self.buffer, dtype=np.float32)
        seq_data = np.expand_dims(seq_data.T, axis=0)
        
        predictions = []
        for group in self.trgt_group_names:
            if group in self.tcn_models:
                # Scale the sequence data for the current group.
                scaled_seq_data, _ = self._standard_scale_features(seq_data, scaler=self.ftr_scalers[group])
                with torch.no_grad():
                    inp_tensor = torch.tensor(scaled_seq_data, dtype=torch.float32, device=self.device)
                    pred_dict = self.tcn_models[group](inp_tensor)
                
                group_preds = []
                for var in self.trgt_vars[group]:
                    group_preds.append(pred_dict[var].item())
                group_preds = np.array(group_preds, dtype=np.float32)
                group_preds = self.trgt_scalers[group].inverse_transform(group_preds.reshape(1, -1)).flatten()
                predictions.append(group_preds)
        
        # Concatenate predictions from all groups that have a loaded model.
        predictions = np.concatenate(predictions, axis=0)
        return predictions
        
    def _aug_nearest_hist_psd (self, predicted_state):
        """
        1) Clip the predicted state values (for the first five target groups) to be within 
           the observation_space bounds.
        2) Compute the Euclidean distance between the clipped predicted subset (for these vars)
           and each row of the historical data (restricted to those same variables).
        3) Find the full historical state corresponding to the closest row.
        4) Replace in that historical state the values for the first five groups with the clipped predicted values.
        5) Return the resulting full state.
        """
        # Clip predicted state for the variables in self.euc_state_vars.
        clipped_predicted = np.empty_like(predicted_state)
        for j, var in enumerate(self.euc_state_vars):
            # Get the index of the variable in the full state vector.
            idx = self.state_vars.index(var)
            low_bound = self.observation_space.low[idx]
            high_bound = self.observation_space.high[idx]
            clipped_predicted[j] = np.clip(predicted_state[j], low_bound, high_bound)
        
        # Compute the Euclidean distances between the clipped predicted subset and the historical subset.
        hist_states_subset = self.data[self.euc_state_vars].to_numpy(dtype=np.float32)
        distances = np.linalg.norm(hist_states_subset - clipped_predicted, axis=1)
        min_idx = np.argmin(distances)
        
        # Retrieve the full historical state corresponding to the nearest row.
        nearest_full = self.data.loc[min_idx, self.state_vars].to_numpy(dtype=np.float32)
        
        # Replace the subset corresponding to the first five target groups with the clipped predictions.
        full_state = np.copy(nearest_full)
        for j, var in enumerate(self.euc_state_vars):
            idx = self.state_vars.index(var)
            full_state[idx] = clipped_predicted[j]
        
        return full_state

    # def switch_mode_and_reset (self):
    #     print(f"Running mode before switch: {'Real-Time' if self.real_time_mode else 'Simulation'}")
    #     self.real_time_mode = not self.real_time_mode
    #     print(f"Current running mode: {'Real-Time' if self.real_time_mode else 'Simulation'}")
    #     return self.reset()
    
    def set_to_real_time (self):
        self.real_time_mode = True
        print(f"Current running mode: {'Real-Time' if self.real_time_mode else 'Simulation'}")
        return self.reset()
    
    def set_to_simulation (self):
        self.real_time_mode = False
        print(f"Current running mode: {'Real-Time' if self.real_time_mode else 'Simulation'}")
        return self.reset()    

    def set_real_time_state(self, plc_states, wipware_states):
        assert self.real_time_mode, "Should not be setting real-time state when in simulated mode!"
        
        # Don't set the state unless both plc and wipware states are provided
        if plc_states == {} or wipware_states == {}:
            return self.real_time_state
        
        state_list =  [plc_states['_output_material_mc'],
                    plc_states['_output_material_temp'],
                    plc_states['_input_material_mc'],
                    plc_states['_input_material_temp'],
                    plc_states['_dryer_bed_temp'],
                    plc_states['_vibe_exhaust_temp_f'],
                    plc_states['_master_cont_mc_pv'],
                    plc_states['_master_cont_mc_sp'],
                    wipware_states['Distance'],
                    wipware_states['Particles'],
                    wipware_states['X50'],
                    wipware_states['Xc'],
                    wipware_states['D01'],
                    wipware_states['D05'],
                    wipware_states['D10'],
                    wipware_states['D20'],
                    wipware_states['D25'],
                    wipware_states['D50'],
                    wipware_states['D75'],
                    wipware_states['D80'],
                    wipware_states['D90'],
                    wipware_states['D95'],
                    wipware_states['D99'],
                    wipware_states['3 mm'],
                    wipware_states['3.5 mm'],
                    wipware_states['4 mm'],
                    wipware_states['5 mm'],
                    wipware_states['6 mm'],
                    wipware_states['7 mm'],
                    wipware_states['8 mm'],
                    wipware_states['9 mm'],
                    wipware_states['10 mm'],
                    wipware_states['11 mm'],
                    wipware_states['12 mm'],
                    wipware_states['13 mm'],
                    wipware_states['14 mm']]

        self.real_time_state = np.array(state_list)
        return self.real_time_state
    