import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim_interface_exp1 import SimInterface
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# sim config
NUM_NODES = 1024
DEFAULT_POWER_NODE = 750
MAX_POWER_NODE = 850
MAX_POWER_CLUSTER = NUM_NODES * MAX_POWER_NODE
POWER_CAP_CLUSTER = 765000  # 90%
MAX_STRESS = 100 # ?

# RL config
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 10_000  # Save model every 10.000 timesteps


class PowercapEnv(gym.Env):
    """
    Custom Gym environment for powercap allocation in a cluster.
    Observation: powercap_status_t packed into numpy array.
    Action: allocate extra_power to greedy nodes + cluster percentage.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator):

        LIMIT_POWER = max(MAX_POWER_CLUSTER, POWER_CAP_CLUSTER)   # considered that powercap > total power ??

        super(PowercapEnv, self).__init__()
        self.simulator = simulator  # external process interface

        # Define observation space:
        # [total_nodes, idle_nodes, released, requested, total_idle_power,
        #  current_power, total_powercap, num_greedy,
        #  greedy_nodes_indices (max 1024),
        #  greedy_bytes(requested, stress, extra_power) for each]
        obs_low = np.zeros(8 + NUM_NODES + NUM_NODES * 3, dtype=np.float32)
        obs_high = np.concatenate([
            np.array([
                    NUM_NODES,              # num of total_nodes
                    NUM_NODES,              # num of idle_nodes
                    LIMIT_POWER,             # accumulated released power in last T1 
                    LIMIT_POWER,             # accumulated new_req 
                    LIMIT_POWER,              # Total power allocated to idle nodes
                    LIMIT_POWER,                 # Accumulated power       
                    LIMIT_POWER,                 # Accumulated current powercap limits
                    NUM_NODES               # num of greedy nodes
                    ], dtype=np.float32),
            np.ones(NUM_NODES) * (NUM_NODES-1),                                  # array([1023., 1023., 1023., ..., 1023.]) lungo 1024
            np.tile([MAX_POWER_NODE, MAX_STRESS, MAX_POWER_NODE], NUM_NODES)
        
        ]).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Define action space:
        
        # typedef struct powercap_opt{
        #     uint32_t num_greedy;        /* Number of greedy nodes */
        #     int32_t *greedy_nodes;      /* List of greedy nodes */
        #     int32_t *extra_power;       /* Extra power received by each greedy node */
        #     uint8_t cluster_perc_power; /* Percentage of total cluster power allocated */
        # } powercap_opt_t;

        # extra_power for each greedy node (int32) + cluster percentage (0-100)
        # We simplify: action = [cluster_perc_power] + extra_power vector length=1024
        # 1 value for cluster_perc_power, from 0 to 100
        # 1024 values for extra_power, from 0 to MAX_EXTRA_POWER (one for every node, 0 if is not used (optionally))

        act_low = np.concatenate([[0], np.zeros(NUM_NODES)])
        act_high = np.concatenate([[100], np.ones(NUM_NODES) * MAX_POWER_NODE])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.simulator.reset()
        self.current_status = state  # <<--- serve per step()
        return self._pack_observation(state), {}


    def _pack_observation(self, status):
        # Flatten powercap_status_t into a 1D numpy array
        header = np.array([
            status.total_nodes,
            status.idle_nodes,
            status.released,
            status.requested,
            status.total_idle_power,
            status.current_power,
            status.total_powercap,
            status.num_greedy
        ], dtype=np.float32)
        # greedy_nodes and data
        gn = np.zeros(1024, dtype=np.float32)
        gb = np.zeros((1024, 3), dtype=np.float32)
        for i, nid in enumerate(status.greedy_nodes[:status.num_greedy]):
            gn[i] = nid
            d = status.greedy_data[i]
            gb[i] = [d.requested, d.stress, d.extra_power]
        flat = np.concatenate([header, gn, gb.flatten()])
        return flat

    
    def step(self, action):
        
        cluster_perc = int(action[0])
        extra = action[1:].astype(np.int32)

        opt = {
            'num_greedy': self.current_status.num_greedy,
            'greedy_nodes': self.current_status.greedy_nodes,
            'extra_power': extra[:self.current_status.num_greedy].tolist(),
            'cluster_perc_power': cluster_perc
        }

        # We ask to the SimInterface a new state
        status = self.simulator.step(opt)
        self.current_status = status
        obs = self._pack_observation(status)
        reward = status.metric_value
        done = status.done
        truncated = status.truncated

        return obs, reward, done, truncated, {}


    def render(self, mode='human'):
        pass

    def close(self):
        self.simulator.close()



# Training script
if __name__ == '__main__':

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    sim = SimInterface(
                num_nodes=NUM_NODES,
                max_watt_per_node=MAX_POWER_NODE,
                cluster_max_power=MAX_POWER_CLUSTER,
                min_node_power=DEFAULT_POWER_NODE
                )
    #env = PowercapEnv(sim)
    env = DummyVecEnv([lambda: PowercapEnv(sim)])
    
    # Upload VecNormalize if exsist
    if os.path.exists(VEC_PATH):
        print("[Main] Carico VecNormalize esistente...")
        env = VecNormalize.load(VEC_PATH, env)
    else:
        print("[Main] Crete new VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)


    os.makedirs(MODEL_DIR, exist_ok=True)

    while True:
        try:

            # If exsist a model, we continue from this one
            if os.path.exists(MODEL_PATH + ".zip"):
                print("[Main] Upload  Model from Model Path")
                model = PPO.load(MODEL_PATH, env=env, device="cpu", tensorboard_log="./logs/")
            else:
                print("[Main] New Model creation")
                model = PPO(
                    "MlpPolicy", 
                    env, verbose=1, 
                    device="cpu", 
                    tensorboard_log="./logs/",
                    n_steps= 8000,                # every 2 simulation. default 2048
                    batch_size=32                 # def 64
                    #learning_rate = 3e-5         # def 3e-4
                    )

            # Callback to save the checkpoint
            checkpoint_callback = CheckpointCallback(
                save_freq=SAVE_FREQ,
                save_path=MODEL_DIR,
                name_prefix="ppo_powercap",
                save_replay_buffer=True,
                save_vecnormalize=True
            )

            # Training
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=checkpoint_callback
            )

            # Save final model trained
            model.save(MODEL_PATH)
            env.save(VEC_PATH)

        except KeyboardInterrupt:
            print("\n[Main] Ctrl+C Interruption, save model and exit")
            model.save(MODEL_PATH)
            env.save(VEC_PATH)
            break
        except Exception as e:
            print("[Main] Unexpected Error:", e)
            model.save(MODEL_PATH)
            env.save(VEC_PATH)
        finally:
            pass
            # print("[Main] Reload and start the simulation again")