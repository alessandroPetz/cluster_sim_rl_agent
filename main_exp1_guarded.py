# RL instantaneous power-cap safeguard (experimental):
# - projects PPO allocations onto the global *allocated powercap* budget
# - keeps the EAR first greedy sentinel at zero
# - does not change the PPO reward or workload semantics
# Requires the accompanying EAR patch for strict simulator enforcement.

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim_interface_exp1_guarded import SimInterface
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch


# sim config

#  CONFIGUARE LA SIMULAZIONE QUI E EAR.CONF (TIPO POWERCAP E POWER SETTIMNG)


NUM_JOBS = 2000
NUM_NODES = 1024
WORKLOAD_VERSION = 95
POWERCAP_PERCENTAGE = 0.8
MAX_POWER_NODE = 850
DEFAULT_POWER_NODE = MAX_POWER_NODE * POWERCAP_PERCENTAGE   # 850 * 0.8 = 680; 850 * 0.9 = 765

MAX_POWER_CLUSTER = NUM_NODES * MAX_POWER_NODE

POWER_CAP_CLUSTER =  MAX_POWER_CLUSTER * POWERCAP_PERCENTAGE # 90%
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

        act_low = np.concatenate([[0], -np.ones(NUM_NODES)*MAX_POWER_NODE])
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
        gn = np.zeros(NUM_NODES, dtype=np.float32)
        gb = np.zeros((NUM_NODES, 3), dtype=np.float32)
        for i, nid in enumerate(status.greedy_nodes[:status.num_greedy]):
            gn[i] = nid
            d = status.greedy_data[i]
            gb[i] = [d.requested, d.stress, d.extra_power]
        flat = np.concatenate([header, gn, gb.flatten()])
        return flat

    
    def _shield_action(self, action):
        """Project a proposed action onto a CONSERVATIVE ALLOCATED-cap budget.

        This constrains sum(node powercaps), not an exact forecast of power.
        It requires a separate Cluster Sim C-side check for a strict guarantee
        across changes caused by new jobs, timers, and P-state updates.
        """
        s = self.current_status
        budget = int(POWER_CAP_CLUSTER)
        n = int(s.num_greedy)
        if n < 0 or n > NUM_NODES or n != len(s.greedy_nodes) or n != len(s.greedy_data):
            raise RuntimeError("Invalid greedy node count or status vectors")

        # Fail closed if the reported observed power already violates the cap.
        # At this point, an earlier simulator event may have caused the violation.
        if int(s.current_power) > budget:
            raise RuntimeError(
                f"Observed cluster power {s.current_power} W exceeds {budget} W; "
                "enforce the cap inside Cluster Sim, not only in Python"
            )

        proposed = np.asarray(action, dtype=np.float64)
        if proposed.ndim != 1 or len(proposed) != 1 + NUM_NODES:
            raise ValueError("Incorrect action shape")
        proposed = np.nan_to_num(proposed, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.rint(proposed[1:1+n]).astype(np.int64)

        already_granted = np.array(
            [max(0, int(g.extra_power)) for g in s.greedy_data], dtype=np.int64
        )
        requested = np.array(
            [max(0, int(g.requested)) for g in s.greedy_data], dtype=np.int64
        )
        # For this model, a node's default allocation is DEFAULT_POWER_NODE.
        # Its already granted extra allocation cannot be removed more than once.
        minimum_delta = -already_granted
        maximum_delta = np.minimum(
            requested,
            np.maximum(0, MAX_POWER_NODE - int(DEFAULT_POWER_NODE) - already_granted),
        )
        bounded = np.clip(raw, minimum_delta, maximum_delta)
        # Do not zero index 0: the companion C fix maps it to its real node.

        # Important: reserved is the powercap that fill_powercap_status()
        # automatically reserves when a running node was below the default.
        # 'released' is only a POTENTIAL release; do not count it as free
        # until it is reflected in total_powercap.
        baseline = int(s.total_powercap) + max(0, int(s.requested))
        delta = np.minimum(bounded, 0)  # start by accepting requested reductions
        deficit = baseline + int(delta.sum()) - budget

        # If the pre-existing allocation is over budget, reclaim additional
        # previously granted extra power where possible; never go below default.
        if deficit > 0:
            for i in range(n):
                can_reclaim = int(delta[i] - minimum_delta[i])
                reclaim = min(deficit, can_reclaim)
                delta[i] -= reclaim
                deficit -= reclaim
                if deficit <= 0:
                    break

        if deficit > 0:
            raise RuntimeError(
                f"Cannot enforce {budget} W: baseline allocation={baseline} W, "
                f"available reclaim insufficient by {deficit} W. "
                "Global cap must be enforced in Cluster Sim C code."
            )

        # Allocate positive requests only from genuinely unallocated power.
        remaining = budget - baseline - int(delta.sum())
        for i in range(n):
            if bounded[i] > 0 and remaining > 0:
                granted = min(int(bounded[i]), remaining)
                delta[i] = granted
                remaining -= granted

        allocated_after = baseline + int(delta.sum())
        assert allocated_after <= budget, (allocated_after, budget)
        info = {
            "power_budget_w": budget,
            "allocated_before_w": int(s.total_powercap),
            "reserved_default_w": max(0, int(s.requested)),
            "allocated_after_upper_bound_w": allocated_after,
            "proposed_sum_delta_w": int(raw.sum()),
            "applied_sum_delta_w": int(delta.sum()),
            "action_was_modified": not np.array_equal(raw, delta),
        }
        return delta.astype(np.int32).tolist(), info

    def step(self, action):
        # SAFETY SHIELD: project BEFORE sending to the external simulator.
        safe_extra, shield_info = self._shield_action(action)
        opt = {
            'num_greedy': self.current_status.num_greedy,
            'greedy_nodes': self.current_status.greedy_nodes,
            'extra_power': safe_extra,
            # Never accept the agent's proposed cluster budget.
            'cluster_perc_power': int(round(POWERCAP_PERCENTAGE * 100)),
        }
        status = self.simulator.step(opt)
        self.current_status = status
        obs = self._pack_observation(status)
        reward = status.metric_value
        done = status.done
        truncated = status.truncated

        # Diagnostic only; cannot undo an already simulated power excursion.
        shield_info["observed_power_after_w"] = int(status.current_power)
        shield_info["observed_violation"] = (not done and
            int(status.current_power) > int(POWER_CAP_CLUSTER))
        return obs, reward, done, truncated, shield_info


    def render(self, mode='human'):
        pass

    def close(self):
        self.simulator.close()



# Training script
if __name__ == '__main__':

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    sim = SimInterface(
                num_jobs=NUM_JOBS,
                num_nodes=NUM_NODES,
                workload_version=WORKLOAD_VERSION,
                max_watt_per_node=MAX_POWER_NODE,
                cluster_max_power=MAX_POWER_CLUSTER,
                min_node_power=DEFAULT_POWER_NODE,
                power_budget_w=POWER_CAP_CLUSTER
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
                policy_kwargs = dict(
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                    activation_fn=torch.nn.Tanh
                )
                # OK per 6080 e 6090
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    device="cpu",
                    tensorboard_log="./logs/",
                    n_steps=8000,
                    batch_size=32,
                    policy_kwargs=policy_kwargs
                )
                # print("[Main] New Model creation")
                # model = PPO(
                #     "MlpPolicy", 
                #     env, verbose=1, 
                #     device="cpu", 
                #     tensorboard_log="./logs/",
                #     n_steps= 8000,                # every 2 simulation. default 2048
                #     batch_size=32                 # def 64
                #     learning_rate = 3e-5         # def 3e-4
                #     )

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