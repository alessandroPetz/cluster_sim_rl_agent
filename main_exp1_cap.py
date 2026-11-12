import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim_interface_exp1_cap import SimInterface
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
import torch


# sim config

#  CONFIGUARE LA SIMULAZIONE QUI E EAR.CONF (TIPO POWERCAP E POWER SETTIMNG)


NUM_JOBS = 2000
NUM_NODES = 1024
WORKLOAD_VERSION = 60
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
        gn = np.zeros(NUM_NODES, dtype=np.float32)
        gb = np.zeros((NUM_NODES, 3), dtype=np.float32)
        for i, nid in enumerate(status.greedy_nodes[:status.num_greedy]):
            gn[i] = nid
            d = status.greedy_data[i]
            gb[i] = [d.requested, d.stress, d.extra_power]
        
        flat = np.concatenate([header, gn, gb.flatten()])

        if not np.all(np.isfinite(flat)):
            bad_indices = np.where(~np.isfinite(flat))[0]

            raise ValueError(
                f"NaN/Inf in observation. "
                f"Indices: {bad_indices[:20]}, "
                f"values: {flat[bad_indices[:20]]}"
            )

        return flat

    
    def _build_feasible_action(self, action):
        """Build EAR deltas from PPO preferences, without a post-action shield.

        PPO action space is unchanged: [cluster_perc, 1024 nonnegative values].
        Each value now controls the desired EXTRA-power allocation to a greedy
        node. The resulting deltas may be negative to return previously granted
        extra power. Assumes EAR applies extra_power as a delta and greedy[0]
        is mapped correctly (EAR first-greedy v2 fix).
        """
        s = self.current_status
        n = s.num_greedy
        budget = int(POWER_CAP_CLUSTER)
        node_headroom = int(MAX_POWER_NODE - DEFAULT_POWER_NODE)

        previous_extra = np.array(
            [max(0, int(g.extra_power)) for g in s.greedy_data[:n]],
            dtype=np.int64
        )
        requested = np.array(
            [max(0, int(g.requested)) for g in s.greedy_data[:n]],
            dtype=np.int64
        )

        # The status may report nodes being restored to their default cap:
        # 'requested' at cluster level accounts for this mandatory allocation.
        # Remove only already-granted EXTRA allocations from the baseline.
        fixed_allocation = (
            int(s.total_powercap)
            + max(0, int(s.requested))
            - int(previous_extra.sum())
        )
        free_for_extra = budget - fixed_allocation

        if free_for_extra < 0:
            raise RuntimeError(
                f"No feasible RL action: mandatory allocation "
                f"{fixed_allocation} W exceeds global budget {budget} W. "
                "This cannot be fixed by redistributing greedy extra power."
            )

        # Each greedy node may keep its existing extra and receive up to its
        # unsatisfied request, without exceeding MAX_POWER_NODE.
        max_extra = np.minimum(node_headroom, previous_extra + requested)

        # Same PPO outputs, interpreted as allocation intensities instead of
        # integer watt deltas. Small positive outputs remain meaningful.
        intensity = np.nan_to_num(
            np.asarray(action[1:1 + n], dtype=np.float64),
            nan=0.0, posinf=MAX_POWER_NODE, neginf=0.0
        )
        intensity = np.clip(intensity, 0.0, MAX_POWER_NODE)
        desired = np.rint(
            max_extra * (-np.expm1(-intensity))
        ).astype(np.int64)

        # Allocate at most the available watts. When there are more requests
        # than power, proportionally distribute the available power using
        # integer largest-remainder allocation (sum never exceeds the cap).
        total_desired = int(desired.sum())
        if total_desired > free_for_extra:
            shares = desired.astype(np.float64) * (
                free_for_extra / total_desired
            )
            target = np.floor(shares).astype(np.int64)
            remaining = free_for_extra - int(target.sum())
            if remaining:
                order = np.argsort(-(shares - target), kind='stable')
                target[order[:remaining]] += 1
        else:
            target = desired

        # EAR expects DELTAS, not final extra-power allocations.
        delta = target - previous_extra
        return {
            'num_greedy': n,
            'greedy_nodes': s.greedy_nodes,
            'extra_power': delta.astype(np.int32).tolist(),
            'cluster_perc_power': int(action[0])
        }

    def step(self, action):

        # rho = 100%
        action = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)


        opt = self._build_feasible_action(action)

        # print("\n--- ACTION DEBUG ---")
        # print("num_greedy:", opt["num_greedy"])
        # print("greedy_nodes:", opt["greedy_nodes"][:20])
        # print("extra_power:", opt["extra_power"][:20])
        # print("sum extra_power:", sum(opt["extra_power"]))
        # print("--------------------")

        # We ask to the SimInterface a new state
        status = self.simulator.step(opt)
        self.current_status = status
        obs = self._pack_observation(status)
        reward = status.metric_value

        # controllo che vada bene la obs
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0]

            raise RuntimeError(
                f"RAW OBSERVATION CONTAINS NaN/Inf. "
                f"indices={bad[:20]}, "
                f"values={obs[bad[:20]]}"
            )

        if not np.isfinite(reward):
            raise RuntimeError(
                f"RAW REWARD CONTAINS NaN/Inf: {reward}"
            )

        done = status.done
        truncated = status.truncated

        reward = status.metric_value

        if not np.isfinite(reward):
            raise ValueError(
                f"Invalid reward: {reward}"
            )
        return obs, reward, done, truncated, {}


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
                power_budget_w=int(POWER_CAP_CLUSTER)
                )
    #env = PowercapEnv(sim)
    env = DummyVecEnv([lambda: PowercapEnv(sim)])

    env = VecCheckNan(
        env,
        raise_exception=True,
        warn_once=False,
        check_inf=True
    )
        
    # Upload VecNormalize if exsist
    if os.path.exists(VEC_PATH):
        print("[Main] Carico VecNormalize esistente...")
        env = VecNormalize.load(VEC_PATH, env)

        def check_finite(name, value):
            value = np.asarray(value)

            if not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"VecNormalize corrupted: {name} contains NaN/Inf"
                )

        check_finite("obs_rms.mean", env.obs_rms.mean)
        check_finite("obs_rms.var", env.obs_rms.var)
        check_finite("ret_rms.mean", env.ret_rms.mean)
        check_finite("ret_rms.var", env.ret_rms.var)

        print("[DEBUG] VecNormalize statistics are finite")

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

                # controllo che non sia corrotto da nan/inf
                for name, param in model.policy.named_parameters():
                    if not torch.isfinite(param).all():
                        raise RuntimeError(
                            f"MODEL ALREADY CORRUPTED: NaN/Inf in {name}"
                        )

                print("[DEBUG] Model parameters are finite")

            else:
                print("[Main] New Model creation")
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=torch.nn.Tanh
                )
                # OK per 6080 e 6090
                SEED = 0
                # model = PPO(
                #     "MlpPolicy",
                #     env,
                #     verbose=1,
                #     device="cpu",
                #     tensorboard_log="./logs/",
                #     n_steps=8000,
                #     batch_size=32,
                #     policy_kwargs=policy_kwargs,
                #     seed=SEED
                # )
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    device="cpu",
                    tensorboard_log="./logs/",

                    n_steps=2048,
                    batch_size=256,
                    learning_rate=1e-4,
                    n_epochs=10,

                    gamma=0.99,
                    gae_lambda=0.95,

                    clip_range=0.2,
                    target_kl=0.03,
                    max_grad_norm=0.5,

                    policy_kwargs=policy_kwargs,
                    seed=SEED
                )

                # model = PPO(
                #     "MlpPolicy",
                #     env,
                #     verbose=1,
                #     device="cpu",
                #     tensorboard_log="./logs/",
                #     n_steps=8000,
                #     batch_size=250,
                #     learning_rate=1e-5,
                #     n_epochs=5,
                #     clip_range=0.2,
                #     target_kl=0.05,
                #     max_grad_norm=0.5,
                #     policy_kwargs=policy_kwargs,
                #     seed=SEED
                # )

                    

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
                callback=checkpoint_callback,
                reset_num_timesteps=False
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
            raise # temèporaneo
            # model.save(MODEL_PATH)
            # env.save(VEC_PATH)
        finally:
            pass
            # print("[Main] Reload and start the simulation again")