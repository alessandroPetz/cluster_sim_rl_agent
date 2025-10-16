import gymnasium as gym
import numpy as np
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sim_interface_exp2 import SimInterface


# ======= CONFIG =======
NUM_NODES = 1024
DEFAULT_POWER_NODE = 750
MAX_POWER_NODE = 850
MAX_POWER_CLUSTER = NUM_NODES * MAX_POWER_NODE
POWER_CAP_CLUSTER = 765000  # 90%
MAX_STRESS = 100

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 10_000
WORKLOAD_TO_TEST=5


# ======= CALLBACK CON PAUSA E TEST =======
class TrainPauseCallback(BaseCallback): 
    """ Ogni train_interval step, il modello esegue una breve fase di test (senza aggiornare i pesi) su un workload mai visto (es. workload 5). """
    def __init__(self, env, train_interval=16000, test_steps=4000, verbose=1): 
        super().__init__(verbose) 
        self.env = env 
        self.train_interval = train_interval 
        self.test_steps = test_steps 
        self.last_pause = 0 
    
    def _on_step(self) -> bool: 
        if self.num_timesteps % 500 == 0: 
            print(f"[Main] Step globale = {self.num_timesteps}") 
            
        if (self.num_timesteps - self.last_pause) >= self.train_interval: 
            self.last_pause = self.num_timesteps 
            self._run_test_phase() 
        
        return True
    
    def _run_test_phase(self):
        print(f"\n[TrainPause] Avvio fase di test a step {self.num_timesteps}...")

        # 🔹 Ottieni riferimento all'unico env reale
        v = getattr(self.env, "venv", self.env)
        envs_list = getattr(v, "envs", [v])

        # 🔹 Imposta modalità test
        for e in envs_list:
            e.test_mode = True
            e.workload_to_test = 5

        # 🔹 Disabilita aggiornamento statistiche di VecNormalize
        if hasattr(self.env, "training"):
            self.env.training = False

        # 🔹 Reset completo (mantiene la socket attiva)
        print("[TrainPause] Reset dell'ambiente per la fase di test...")
        obs = self.env.reset()
        total_reward = 0.0
        steps = 0

        print("[TrainPause] Inizio esecuzione test...")

        while True:
            try:
                # Predici azione (senza learning)
                action, _ = self.model.predict(obs, deterministic=True)

                # Compatibilità Gym/Gymnasium
                step_out = self.env.step(action)
                if len(step_out) == 5:
                    obs, reward, done, truncated, info = step_out
                else:
                    obs, reward, done, info = step_out
                    truncated = np.array([False])

                total_reward += np.mean(reward)
                steps += 1

                if steps % 500 == 0:
                    print(f"[TrainPause] Step test = {steps}")

                # Fine episodio
                if getattr(done, "any", lambda: done)() or getattr(truncated, "any", lambda: truncated)():
                    print(f"[TrainPause] Test terminato dopo {steps} step.")
                    break

                if steps >= self.test_steps:
                    print(f"[TrainPause] Fine test (raggiunto limite {self.test_steps}).")
                    break

            except Exception as e:
                print(f"[TrainPause] ⚠️ Errore durante test step: {e}")
                break

        avg_reward = total_reward / max(1, steps)
        print(f"[TrainPause] Ricompensa media test = {avg_reward:.4f}")

        # 🔹 Torna in modalità training
        for e in envs_list:
            e.test_mode = False
            e.workload_to_test = None

        # 🔹 Riattiva il VecNormalize in modalità training
        if hasattr(self.env, "training"):
            self.env.training = True

        # 🔹 Reset completo dell'ambiente per far ripartire il training
        print("[TrainPause] Reset simulatore e ritorno in training...")
        try:
            obs = self.env.reset()
        except Exception as e:
            print(f"[TrainPause] ⚠️ Errore nel reset finale: {e}")

        self.logger.record("test/mean_reward", avg_reward)

    # def _run_test_phase(self):
    #     print(f"\n[TrainPause] Avvio fase di test a step {self.num_timesteps} (no learning)...")

    #     # 🔹 Ottieni gli env interni
    #     v = getattr(self.env, "venv", self.env)
    #     envs_list = getattr(v, "envs", [v])

    #     # 🔹 Disabilita aggiornamento statistiche VecNormalize
    #     if hasattr(self.env, "training"):
    #         self.env.training = False

    #     # 🔹 Attiva modalità test negli env esistenti
    #     for i, env_instance in enumerate(envs_list):
    #         env_instance.test_mode = True
    #         env_instance.workload_to_test = WORKLOAD_TO_TEST
    #         try:
    #             print(f"[TrainPause] Env {i}: reset simulatore in modalità TEST...")
    #             env_instance.simulator.reset(test_mode=True, workload_to_test=5)
    #             time.sleep(0.5)
    #         except Exception as e:
    #             print(f"[TrainPause] ⚠️ Errore reset simulatore env {i}: {e}")

    #     # 🔹 Reset ambiente Gym (sincronizza osservazioni)
    #     obs = self.env.reset()
    #     total_reward = 0.0
    #     n_steps = 0

    #     print("[TrainPause] Inizio fase di test...")
    #     start_time = time.time()

    #     while n_steps < self.test_steps:
    #         try:
    #             action, _ = self.model.predict(obs, deterministic=True)
    #             obs, reward, done, info = self.env.step(action)
    #             total_reward += np.mean(reward)
    #             n_steps += 1

    #             if n_steps % 100 == 0:
    #                 print(f"[TrainPause] Step test {n_steps}/{self.test_steps} — reward={np.mean(reward):.3f}")

    #             # 🔹 Se episodio terminato, resetta e prosegui
    #             if getattr(done, "any", lambda: done)():
    #                 obs = self.env.reset()

    #         except Exception as e:
    #             print(f"[TrainPause] ⚠️ Errore durante lo step di test: {e}")
    #             break

    #     avg_reward = total_reward / max(1, n_steps)
    #     elapsed = time.time() - start_time
    #     print(f"[TrainPause] Fine test — {n_steps} step, durata {elapsed:.1f}s, ricompensa media={avg_reward:.4f}")

    #     # 🔹 Ripristina modalità training
    #     for i, env_instance in enumerate(envs_list):
    #         env_instance.test_mode = False
    #         # env_instance.workload_to_test = None
    #         try:
    #             print(f"[TrainPause] Env {i}: reset simulatore in modalità TRAIN...")
    #             env_instance.simulator.reset(test_mode=False)
    #             time.sleep(0.5)
    #         except Exception as e:
    #             print(f"[TrainPause] ⚠️ Errore reset simulatore env {i} (train): {e}")

    #     # 🔹 Riattiva normalizzazione
    #     if hasattr(self.env, "training"):
    #         self.env.training = True

    #     # 🔹 Reset finale dell’ambiente (nuovo workload casuale)
    #     print("[TrainPause] Reset finale ambiente per riprendere il training...")
    #     self.env.reset()

    #     # 🔹 Log reward medio
    #     self.logger.record("test/mean_reward", avg_reward)

    # def _run_test_phase(self):
    #         print(f"\n[TrainPause] Avvio fase di test a step {self.num_timesteps}...")

    #         # 🔹 Imposta test_mode
    #         v = getattr(self.env, "venv", self.env)
    #         envs_list = getattr(v, "envs", [v])
    #         for e in envs_list:
    #             e.test_mode = True
    #             e.workload_to_test = 5

    #         # 🔹 Reset completo del simulatore
    #         obs = self.env.reset()
    #         total_reward = 0.0
    #         steps = 0

    #         while True:
    #             action, _ = self.model.predict(obs, deterministic=True)
    #             obs, reward, done, truncated, info = self.env.step(action)
    #             total_reward += np.mean(reward)
    #             steps += 1

    #             if steps % 500 == 0:
    #                 print("[Main] Step test = ",steps)

    #             # 🔹 Se il simulatore segnala fine o truncate, interrompi test
    #             if getattr(done, "any", lambda: done)() or getattr(truncated, "any", lambda: truncated)():
    #                 print(f"[TrainPause] Test terminato dopo {steps} step.")
    #                 break

    #             if steps >= self.test_steps:
    #                 print(f"[TrainPause] Fine test (raggiunto limite {self.test_steps}).")
    #                 break

    #         avg_reward = total_reward / steps
    #         print(f"[TrainPause] Ricompensa media test = {avg_reward:.4f}")

    #         # 🔹 Torna in modalità train
    #         for e in envs_list:
    #             e.test_mode = False
    #             e.workload_to_test = WORKLOAD_TO_TEST

    #         # 🔹 Reset completo del simulatore prima di ripartire col training
    #         print("[TrainPause] Reset simulatore e ritorno in training...")
    #         obs = self.env.reset()

    #         self.logger.record("test/mean_reward", avg_reward)


# ======= ENV PERSONALIZZATO =======
class PowercapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator):
        super(PowercapEnv, self).__init__()
        self.simulator = simulator

        LIMIT_POWER = max(MAX_POWER_CLUSTER, POWER_CAP_CLUSTER)

        obs_low = np.zeros(8 + NUM_NODES + NUM_NODES * 3, dtype=np.float32)
        obs_high = np.concatenate([
            np.array([
                NUM_NODES, NUM_NODES, LIMIT_POWER, LIMIT_POWER,
                LIMIT_POWER, LIMIT_POWER, LIMIT_POWER, NUM_NODES
            ], dtype=np.float32),
            np.ones(NUM_NODES) * (NUM_NODES - 1),
            np.tile([MAX_POWER_NODE, MAX_STRESS, MAX_POWER_NODE], NUM_NODES)
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        act_low = np.concatenate([[0], np.zeros(NUM_NODES)])
        act_high = np.concatenate([[100], np.ones(NUM_NODES) * MAX_POWER_NODE])
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Default mode
        self.test_mode = False
        self.workload_to_test = WORKLOAD_TO_TEST

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.simulator.reset(test_mode=self.test_mode, workload_to_test=self.workload_to_test)
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
        gn = np.zeros(1024, dtype=np.float32)
        gb = np.zeros((1024, 3), dtype=np.float32)
        for i, nid in enumerate(status.greedy_nodes[:status.num_greedy]):
            gn[i] = nid
            d = status.greedy_data[i]
            gb[i] = [d.requested, d.stress, d.extra_power]
        return np.concatenate([header, gn, gb.flatten()])

    def step(self, action):
        cluster_perc = int(action[0])
        extra = action[1:].astype(np.int32)

        opt = {
            'num_greedy': self.current_status.num_greedy,
            'greedy_nodes': self.current_status.greedy_nodes,
            'extra_power': extra[:self.current_status.num_greedy].tolist(),
            'cluster_perc_power': cluster_perc
        }

        status = self.simulator.step(opt, test_mode=self.test_mode, workload_to_test=self.workload_to_test)
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


# ======= FUNZIONE PER CREARE UN ENV =======
def make_env():
    sim_local = SimInterface(
        num_nodes=NUM_NODES,
        max_watt_per_node=MAX_POWER_NODE,
        cluster_max_power=MAX_POWER_CLUSTER,
        min_node_power=DEFAULT_POWER_NODE
    )
    return PowercapEnv(sim_local)


# ======= MAIN TRAINING =======
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = DummyVecEnv([make_env])

    # Carica o crea VecNormalize
    if os.path.exists(VEC_PATH):
        print("[Main] Carico VecNormalize esistente...")
        env = VecNormalize.load(VEC_PATH, env)
    else:
        print("[Main] Creo nuovo VecNormalize...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Assicura settaggio iniziale
    venv = getattr(env, "venv", env)
    venv.set_attr("test_mode", False)
    venv.set_attr("workload_to_test", WORKLOAD_TO_TEST)

    while True:
        try:
            if os.path.exists(MODEL_PATH + ".zip"):
                print("[Main] Carico modello esistente...")
                model = PPO.load(MODEL_PATH + ".zip", env=env, device="cpu", tensorboard_log="./logs/")
            else:
                print("[Main] Creo nuovo modello...")
                model = PPO("MlpPolicy",
                            env,
                            verbose=1,
                            device="cpu",
                            tensorboard_log="./logs/",
                            n_steps=8000,  #8000
                            batch_size=32) #32

            checkpoint_callback = CheckpointCallback(
                save_freq=SAVE_FREQ,
                save_path=MODEL_DIR,
                name_prefix="ppo_powercap",
                save_replay_buffer=True,
                save_vecnormalize=True
            )

            pause_callback = TrainPauseCallback(env, train_interval=4000, test_steps=4000)

            print("[Main] Avvio training...")
            model.learn(total_timesteps=TOTAL_TIMESTEPS,
                        callback=[checkpoint_callback, pause_callback])

            model.save(MODEL_PATH)
            env.save(VEC_PATH)

        except KeyboardInterrupt:
            print("\n[Main] Interrotto da tastiera. Salvo modello ed esco.")
            model.save(MODEL_PATH)
            env.save(VEC_PATH)
            break
        except Exception as e:
            print("[Main] Errore imprevisto:", e)
            model.save(MODEL_PATH)
            env.save(VEC_PATH)
        finally:
            print("[Main] Ricarico e riparto con la simulazione.")
