import os
import time
import argparse
from tkinter.font import families
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviaryQ3 import HoverAviaryQ3
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_MODEL_PATH = '/Users/jinceyang/Desktop/codebase/nuscourse/nus_ceg_homework/ceg5306/pa3/gym-pybullet-drones/results/save-q3-10.05.2025_20.40.20/best_model.zip'
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
# DEFAULT_ACT = ActionType('one_d_rpm')
DEFAULT_ACT = ActionType('rpm')

def play(model_path=DEFAULT_MODEL_PATH, gui=DEFAULT_GUI):
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = PPO.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")


    env = HoverAviaryQ3(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, record=False)
    

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=1,
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()
        state_vec = env._getDroneStateVector(0)
    
        logger.log(
            drone=0,
            timestamp=i/env.CTRL_FREQ,
            state=state_vec,
            control=np.zeros(12)
        )
           
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    logger.plot()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    args = parser.parse_args()

    play(**vars(args))
