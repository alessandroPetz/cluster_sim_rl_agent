#
# Experiment 2: leave-one-case-out generalization.
# We train the model on 5 of the 6 workload/powercap cases and test on the remaining one.
#

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sim_interface_exp2_new import SimInterface
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
import torch

# sim config
NUM_JOBS = 2000
NUM_NODES = 1024
MAX_POWER_NODE = 850
MAX_POWER_CLUSTER = NUM_NODES * MAX_POWER_NODE
MAX_STRESS = 100

# Case left out from training and used only for test (1..6).
# The other five cases are selected automatically by SimInterface.
TEST_CASE = 1

# RL config - same compact 6-dimensional strategy used in experiment 1.
# Use new filenames: old experiment-2 checkpoints have a 1025-dimensional action space
# and are NOT compatible with this agent.
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, f"ppo_powercap_exp2_case{TEST_CASE}_vB6")
VEC_PATH = os.path.join(MODEL_DIR, f"vecnormalize_exp2_case{TEST_CASE}_vB6.pkl")
TENSORBOARD_LOG = "./logs/"

ACTION_DIM = 6
BETA_MIN = 0.25
BETA_MAX = 6.0


class PowercapEnv(gym.Env):
    """
    Custom Gym environment for cluster power allocation.
    Observation: powercap_status_t packed into a 1D vector.
    Action: the same compact 6-parameter allocation strategy used in experiment 1.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator):
        super(PowercapEnv, self).__init__()
        self.simulator = simulator

        # Observation bounds must cover both 80% and 90% cases.
        obs_low = np.zeros(8 + NUM_NODES + NUM_NODES * 3, dtype=np.float32)
        obs_high = np.concatenate([
            np.array([
                NUM_NODES,
                NUM_NODES,
                MAX_POWER_CLUSTER,
                MAX_POWER_CLUSTER,
                MAX_POWER_CLUSTER,
                MAX_POWER_CLUSTER,
                MAX_POWER_CLUSTER,
                NUM_NODES
            ], dtype=np.float32),
            np.ones(NUM_NODES) * (NUM_NODES - 1),
            np.tile([MAX_POWER_NODE, MAX_STRESS, MAX_POWER_NODE], NUM_NODES)
        ]).astype(np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Same Version-B action used by experiment 1:
        # action[0] = rho: fraction of feasible EXTRA-power budget to use
        # action[1] = weight for normalized requested power
        # action[2] = weight for normalized stress
        # action[3] = weight for already allocated extra power
        # action[4] = weight for a new greedy node
        # action[5] = concentration/aggressiveness of the priority distribution
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # SimInterface selects one of the five training cases, or TEST_CASE in test mode.
        state = self.simulator.reset()
        self.current_status = state
        return self._pack_observation(state), {}

    def _pack_observation(self, status):
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

    @staticmethod
    def _allocate_capped_weighted(weights, capacities, total_power):
        """Allocate integer watts proportionally, respecting per-node capacities.

        Any watts left because a node reaches its capacity are redistributed among
        the remaining active nodes. The returned vector always satisfies:

            0 <= allocation[i] <= capacities[i]
            sum(allocation) <= total_power
        """
        capacities = np.asarray(capacities, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        target = np.zeros_like(capacities, dtype=np.int64)
        if capacities.size == 0:
            return target

        total_power = int(max(0, total_power))
        total_power = min(total_power, int(capacities.sum()))
        remaining = total_power

        while remaining > 0:
            active = np.flatnonzero(target < capacities)
            if active.size == 0:
                break

            active_weights = weights[active]
            if (
                not np.all(np.isfinite(active_weights))
                or float(active_weights.sum()) <= 0.0
            ):
                active_weights = np.ones(active.size, dtype=np.float64)

            active_weights = active_weights / active_weights.sum()
            raw = remaining * active_weights
            room = capacities[active] - target[active]

            grant = np.minimum(np.floor(raw).astype(np.int64), room)
            granted = int(grant.sum())

            if granted > 0:
                target[active] += grant
                remaining -= granted
                continue

            # If remaining watts are fewer than active nodes, floor(raw) can be
            # all zero. Assign one watt to the highest fractional priorities.
            order = np.argsort(-raw, kind='stable')
            count = min(remaining, active.size)
            chosen = active[order[:count]]
            target[chosen] += 1
            remaining -= count

        return target

    def _build_feasible_action(self, action):
        """Convert the compact Version-B PPO action into feasible EAR deltas.

        The six PPO outputs are:
            0: rho, fraction of feasible extra-power budget to use
            1: requested-power weight
            2: stress weight
            3: previous-extra-power weight
            4: new-greedy-node weight
            5: concentration parameter for the node priority distribution

        All PPO outputs live in [-1, 1]. Only rho is remapped to [0, 1].
        The four feature weights remain signed. The last value is remapped to
        beta in [BETA_MIN, BETA_MAX], where larger beta produces a more
        concentrated allocation.
        """
        s = self.current_status
        n = int(s.num_greedy)
        # Experiment 2 changes case at every reset (80% or 90% cap).
        # Use the values selected by SimInterface for the CURRENT episode.
        budget = int(self.simulator.current_power_budget_w)
        default_power_node = int(self.simulator.current_default_power_node)
        node_headroom = int(MAX_POWER_NODE - default_power_node)

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Version-B action must have shape ({ACTION_DIM},), "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"NaN/Inf in Version-B action: {action}")

        action = np.clip(action, -1.0, 1.0)

        # Global decision: how much of the feasible EXTRA budget should be used?
        rho = 0.5 * (action[0] + 1.0)

        # State-dependent scoring parameters selected by PPO.
        w_requested = action[1]
        w_stress = action[2]
        w_previous_extra = action[3]
        w_new_greedy = action[4]

        # Larger beta => stronger preference for the highest-scoring nodes.
        beta01 = 0.5 * (action[5] + 1.0)
        beta = BETA_MIN + beta01 * (BETA_MAX - BETA_MIN)

        previous_extra = np.array(
            [max(0, int(g.extra_power)) for g in s.greedy_data[:n]],
            dtype=np.int64
        )
        requested = np.array(
            [max(0, int(g.requested)) for g in s.greedy_data[:n]],
            dtype=np.int64
        )
        stress = np.array(
            [max(0, min(MAX_STRESS, int(g.stress))) for g in s.greedy_data[:n]],
            dtype=np.float64
        )

        # Mandatory/base allocation. Existing greedy extra is removed here and
        # reintroduced through target[] below, so PPO can also decide to return it.
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

        # Maximum FINAL extra allocation allowed for each greedy node.
        # requested is interpreted as still-unsatisfied extra demand.
        max_extra = np.minimum(
            node_headroom,
            previous_extra + requested
        ).astype(np.int64)

        # Normalize per-node features to comparable [0, 1] scales.
        denom = float(max(1, node_headroom))
        requested_norm = np.clip(requested / denom, 0.0, 1.0)
        stress_norm = np.clip(stress / float(MAX_STRESS), 0.0, 1.0)
        previous_extra_norm = np.clip(previous_extra / denom, 0.0, 1.0)
        new_greedy = ((requested > 0) & (previous_extra == 0)).astype(np.float64)

        # PPO does not choose per-node watts. It chooses how node features should
        # be valued in this state. The resulting score is converted to a positive
        # priority distribution with a stable softmax.
        scores = (
            w_requested * requested_norm
            + w_stress * stress_norm
            + w_previous_extra * previous_extra_norm
            + w_new_greedy * new_greedy
        )

        if n > 0:
            logits = beta * scores
            logits = logits - np.max(logits)
            priorities = np.exp(logits)
            priorities[max_extra <= 0] = 0.0

            if float(priorities.sum()) <= 0.0:
                priorities = (max_extra > 0).astype(np.float64)
        else:
            priorities = np.empty(0, dtype=np.float64)

        # rho decides HOW MUCH extra power to use. This is deliberately separate
        # from the node priorities, which decide WHERE that power should go.
        available_extra = min(
            int(free_for_extra),
            int(max_extra.sum())
        )
        power_to_allocate = int(np.rint(rho * available_extra))

        target = self._allocate_capped_weighted(
            priorities,
            max_extra,
            power_to_allocate
        )

        # EAR expects DELTAS relative to already allocated extra power.
        delta = target - previous_extra

        positive_delta = np.maximum(delta, 0)

        # check if is bad
        # bad = np.where(positive_delta > requested)[0]

        # if bad.size > 0:
        #     raise RuntimeError(
        #         f"RL OVER-REQUEST: "
        #         f"idx={bad[:20]}, "
        #         f"delta={positive_delta[bad[:20]]}, "
        #         f"requested={requested[bad[:20]]}, "
        #         f"previous={previous_extra[bad[:20]]}, "
        #         f"target={target[bad[:20]]}"
        #     )

        # Do not spend one PPO output on this bookkeeping field. Mimic the hard
        # policy convention: report the fraction of the global budget currently
        # represented by total_powercap. The actual node-level control is delta[].
        if budget > 0:
            cluster_perc_power = int(np.clip(
                np.rint(100.0 * int(s.total_powercap) / budget),
                0,
                100
            ))
        else:
            cluster_perc_power = 0

        # Small diagnostic payload; useful when inspecting learning behavior.
        self.last_action_debug = {
            'rho': float(rho),
            'w_requested': float(w_requested),
            'w_stress': float(w_stress),
            'w_previous_extra': float(w_previous_extra),
            'w_new_greedy': float(w_new_greedy),
            'beta': float(beta),
            'free_for_extra': int(free_for_extra),
            'available_extra': int(available_extra),
            'target_extra': int(target.sum()),
            'previous_extra': int(previous_extra.sum()),
            'delta_extra': int(delta.sum()),
        }

        return {
            'num_greedy': n,
            'greedy_nodes': s.greedy_nodes,
            'extra_power': delta.astype(np.int32).tolist(),
            'cluster_perc_power': cluster_perc_power
        }

    def step(self, action):
        
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
        return obs, reward, done, truncated, dict(self.last_action_debug)


    def render(self, mode='human'):
        pass

    def close(self):
        self.simulator.close()


if __name__ == '__main__':

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Keep the original experiment-2 train/test cycle.
    TRAIN_STEPS = 8_000
    TEST_STEPS = 4_000

    while True:
        try:
            # =========================
            # TRAIN on 5 of the 6 cases
            # =========================
            sim = SimInterface(
                num_jobs=NUM_JOBS,
                num_nodes=NUM_NODES,
                max_watt_per_node=MAX_POWER_NODE,
                cluster_max_power=MAX_POWER_CLUSTER,
                test_case=TEST_CASE
            )

            env = DummyVecEnv([lambda: PowercapEnv(sim)])
            env = VecCheckNan(
                env,
                raise_exception=True,
                warn_once=False,
                check_inf=True
            )

            if os.path.exists(VEC_PATH):
                env = VecNormalize.load(VEC_PATH, env)
            else:
                env = VecNormalize(
                    env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_reward=10.0
                )

            if os.path.exists(MODEL_PATH + ".zip"):
                model = PPO.load(
                    MODEL_PATH,
                    env=env,
                    device="cpu",
                    tensorboard_log=TENSORBOARD_LOG
                )
            else:
                print("[Main] New Model creation")
                policy_kwargs = dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=torch.nn.Tanh
                )

                # Same PPO configuration currently used in experiment 1.
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    device="cpu",
                    tensorboard_log=TENSORBOARD_LOG,
                    n_steps=8000,
                    batch_size=250,
                    learning_rate=1e-5,
                    n_epochs=5,
                    clip_range=0.2,
                    target_kl=0.05,
                    max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs,
                    seed=0
                )

            print("\n[Main] >>> TRAIN PHASE - start rollout <<<")
            sim.set_mode("train")
            model.learn(
                total_timesteps=TRAIN_STEPS,
                reset_num_timesteps=False
            )
            model.save(MODEL_PATH)
            env.save(VEC_PATH)

            # ========================
            # TEST on the excluded case
            # ========================
            print("\n[Main] >>> TEST PHASE - start rollout <<<")

            sim_test = SimInterface(
                num_jobs=NUM_JOBS,
                num_nodes=NUM_NODES,
                max_watt_per_node=MAX_POWER_NODE,
                cluster_max_power=MAX_POWER_CLUSTER,
                test_case=TEST_CASE
            )

            test_env = DummyVecEnv([lambda: PowercapEnv(sim_test)])
            test_env = VecCheckNan(
                test_env,
                raise_exception=True,
                warn_once=False,
                check_inf=True
            )

            if os.path.exists(VEC_PATH):
                test_env = VecNormalize.load(VEC_PATH, test_env)
                test_env.training = False
                test_env.norm_reward = False

            model = PPO.load(MODEL_PATH, env=test_env, device="cpu")

            sim_test.set_mode("test")
            obs = test_env.reset()
            for step in range(TEST_STEPS):
                # Kept as in the original experiment 2 to minimize protocol changes.
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, _ = test_env.step(action)

            print("[Main] >>> TEST PHASE - exit <<<")

        except KeyboardInterrupt:
            print("\n[Main] Ctrl+C Interruption, save model and exit")
            try:
                model.save(MODEL_PATH)
                env.save(VEC_PATH)
            except Exception:
                pass
            break

        except Exception as e:
            print("[Main] Unexpected Error:", e)
            raise
