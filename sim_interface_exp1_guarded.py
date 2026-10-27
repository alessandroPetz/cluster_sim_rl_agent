import subprocess
import json
import os
import time
import glob
import shutil
import csv
import select
import socket
import psutil
from datetime import datetime
from pathlib import Path

# Riprendo le stesse classi dati che avevi nel tuo esempio:
class GreedyBytes:
    def __init__(self, requested, stress, extra_power):
        self.requested = requested
        self.stress = stress
        self.extra_power = extra_power

class PowercapStatus:
    def __init__(self, total_nodes, idle_nodes, released, requested,
                 total_idle_power, current_power, total_powercap,
                 greedy_nodes, greedy_data, metric_value=None, done=False, truncated=False):
        self.total_nodes = total_nodes
        self.idle_nodes = idle_nodes
        self.released = released
        self.requested = requested
        self.total_idle_power = total_idle_power
        self.current_power = current_power
        self.total_powercap = total_powercap
        self.num_greedy = len(greedy_nodes)
        self.greedy_nodes = greedy_nodes
        self.greedy_data = greedy_data
        self.metric_value = metric_value
        self.done = done
        self.truncated = truncated


class SimInterface:
    def __init__(self,
                 num_jobs,
                 num_nodes,
                 workload_version,
                 max_watt_per_node,
                 cluster_max_power,
                 min_node_power,
                 host='localhost', 
                 port=12345,
                 power_budget_w=None
                 ):
        """
        - cmd: percorso (o nome) dell’eseguibile C++, es. "./sim"
        - num_nodes, max_watt_per_node, ecc. sono parametri che manteniamo
          per compatibilità, ma non vengono usati direttamente dal Python.
        """
        
        self.num_jobs=num_jobs
        self.num_nodes = num_nodes
        self.workload_version = workload_version
        self.max_watt_per_node = max_watt_per_node
        self.cluster_max_power = cluster_max_power
        self.min_node_power = min_node_power
        self.host = host
        self.port = port
        self.power_budget_w = power_budget_w

        self.step_counter = 0
        self.status = None
        self.row_summary_file_status = 0


    def kill_all_processes(self):
        print("[SimInterface] Killing old simulation processes: batsim, batsched, eargmd, cluster_sim")

        for proc, name in [
            (getattr(self, "eargmd_proc", None), "eargmd"),
            (getattr(self, "batsim_proc", None), "batsim"),
            (getattr(self, "batsched_proc", None), "batsched"),
            (getattr(self, "cluster_sim_proc", None), "cluster_sim"),
            (getattr(self, "proc", None), "eargmd (agent)")
        ]:
            if proc and proc.poll() is None:  # Se è ancora attivo
                try:
                    #print(f"[SimInterface] Killing {name} (pid={proc.pid})")
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    #print(f"[SimInterface] Kill forzato di {name} (pid={proc.pid})")
                    proc.kill()
                except Exception as e:
                    pass
                    #print(f"[SimInterface] Errore durante kill di {name}: {e}")


        # Here we kill processes of other simulation still alived for some reason    
        # print("[SimInterface] Kill external processes")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for name in ["eargmd","batsim","batsched","cluster_sim"]:
                    if name in proc.info['name'] or any(name in arg for arg in proc.info.get('cmdline', [])):
                        print(f"[SimInterface] Killing {name} (pid={proc.pid})")
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            print(f"[SimInterface] Kill forzato di {name} (pid={proc.pid})")
                            proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def process_summary_and_save_results(self, file_path):

        # Read summary only to check simulation completion
        last_job_id = -1
        last_end_time = None

        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')

            for row in reader:
                last_job_id = row['job_id'].strip()
                last_end_time = float(row['end_time'])

        # Save results only when simulation is completed
        if last_job_id == "IDLE":

            # Find corresponding trace file
            trace_path = file_path.replace("summary.csv", ".csv")

            if not os.path.exists(trace_path):
                raise FileNotFoundError(
                    f"[SimInterface] Trace file not found: {trace_path}"
                )

            # Calculate energy directly from trace
            energy_active = 0.0
            energy_idle = 0.0

            with open(trace_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')

                previous_row = next(reader, None)

                if previous_row is None:
                    raise ValueError("Trace file is empty")

                for row in reader:

                    t_previous = float(previous_row['timestamp'])
                    t_current = float(row['timestamp'])

                    dt = t_current - t_previous

                    if dt < 0:
                        raise ValueError("Invalid trace timestamps")

                    active_power = float(previous_row['active_power'])
                    idle_power = float(previous_row['idle_power'])

                    energy_active += active_power * dt
                    energy_idle += idle_power * dt

                    previous_row = row

            energy_sum = energy_active + energy_idle

            print(f"[SimInterface] Active energy: {energy_active} J")
            print(f"[SimInterface] Idle energy: {energy_idle} J")
            print(f"[SimInterface] Total energy: {energy_sum} J")
            print(f"[SimInterface] Total energy: {energy_sum / 3600000:.2f} kWh")
            print(f"[SimInterface] Last end_time: {last_end_time}")

            # Save results
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            header = ['timestamp', 'filename', 'energy', 'end_time']

            nuova_riga = [
                timestamp,
                trace_path,
                energy_sum,
                last_end_time
            ]

            output_path = "history_summaries/results.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            file_exist = os.path.exists(output_path)

            with open(output_path, 'a', newline='') as file:
                writer = csv.writer(file)

                if not file_exist:
                    writer.writerow(header)

                writer.writerow(nuova_riga)

            print("[SimInterface] Results saved successfully")

    def remove_metric_files_and_save_summary(self):

        print("[SimInterface] Remove metric/trace files and save summary.csv in history folder")

        # move summary file in history folder and remove other files

        current_dir = os.getcwd()                                       # current folder
        history_dir = os.path.join(current_dir, 'history_summaries')    # history filder

        # create history folder 
        if not os.path.exists(history_dir):                             
            os.makedirs(history_dir)

        # Find summary file move it in history folder and 
        # if is completed, save the results in history/results.csv
        for filename in os.listdir(current_dir):
            file_path = os.path.join(current_dir, filename)
            if os.path.isfile(file_path) and 'summary' in filename.lower():

                # process summary csv and in case save results
                self.process_summary_and_save_results(file_path)       

                # move summary in history folder
                dest_path = os.path.join(history_dir, filename)
                shutil.move(file_path, dest_path)

                # save the file with number of active node and energy history
                file_path2 = os.path.join(current_dir, filename.replace("summary", ""))
                dest_path2 = os.path.join(history_dir, filename.replace("summary", ""))
                shutil.move(file_path2, dest_path2)

                print(f"Moved: {filename} → {history_dir}")
        
        
        # Remove other files
        files = (glob.glob("*.csv"))
        try:
            for f in files:
                if f and os.path.exists(f):
                    os.remove(f)
        except:
            pass
            # print("no .csv to remove")
        try:
            files = (glob.glob("*.trace"))
            for f in files:
                if f and os.path.exists(f):
                    os.remove(f)
        except:
            pass
            #print("no .trace tu remove")

    def run_simulation(self):
        
        # To start with the simulation, I have to run 4 external processes

        print("[SimInterface] >>> Start with the simulation processes (eargmd, batsim, batsche, cluster_sim")

        # processes setting
        # Workspace is the folder that cotain the folder that contain the python script
        script_path = Path(__file__).resolve()
        workspace = script_path.parent.parent

        # check if exist tmp folder
        tmp_dir = os.path.join(workspace, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Variabili d'ambiente
        env_eargmd = os.environ.copy()
        env_eargmd["EAR_ETC"] = f"{workspace}/EAR/etc"

        env_cluster_sim = os.environ.copy()
        env_cluster_sim["CLUSTER_SIM_NUM_NODES"] = str(self.num_nodes)
        env_cluster_sim["CLUSTER_SIM_DEF_POWERCAP"] = str(self.min_node_power)
        if self.power_budget_w is not None:
            env_cluster_sim["CLUSTER_SIM_GLOBAL_CAP_W"] = str(int(self.power_budget_w))

        #print("[SimInterface] >>> Start eargmd...")
        self.eargmd_proc = subprocess.Popen(
            [f"{workspace}/source/ear_private/src/global_manager/eargmd"],
            stdout=open(f"{workspace}/tmp/eargmd.log", "w"),
            stderr=subprocess.STDOUT,
            env=env_eargmd
        )

        time.sleep(1)

        #print("[SimInterface] >>> Start batsim...")
        self.batsim_proc = subprocess.Popen(
            [
                "batsim",
                "-p", f"{workspace}/input_files/cluster_{self.num_nodes}nodes.xml",
                "--mmax-workload",
                "-w", f"{workspace}/input_files/workload_{self.num_jobs}jobs_{self.num_nodes}nodes_{self.workload_version}.json",
                "-E"
            ],
            stdout=open(f"{workspace}/tmp/batsim.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1)

        #print("[SimInterface] >>> Start batsched...")
        self.batsched_proc = subprocess.Popen(
            ["batsched", "-v", "easy_bf", "--verbosity=debug"],
            stdout=open(f"{workspace}/tmp/batsched.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1)

        #print("[SimInterface] >>> Start cluster_sim...")
        self.cluster_sim_proc = subprocess.Popen(
            [
                f"{workspace}/source/ear_private/src/tools/cluster_sim",
                "test_tag",
                f"../input_files/power_{self.num_jobs}jobs.txt" 
                
            ],
            env=env_cluster_sim,
            stdout=open(f"{workspace}/tmp/cluster_sim.log", "w"),
            stderr=subprocess.STDOUT,
            cwd = script_path.parent
        )

        print("[SImInterface] >>> All the processes are running.")

    def connect_at_simulation_socket(self, host='localhost', port=12345, timeout=4, delay=0.5):
        """
        Try to connect to the simulation socket repeatedly until the timeout expires.
        """
        start_time = time.time()
        while True:
            try:
                self.cpp_sock = socket.create_connection((host, port))
                self.cpp_file_in = self.cpp_sock.makefile('r')
                self.cpp_file_out = self.cpp_sock.makefile('w')
                print(f"[SimInterface] Connected at the socket {host}:{port}")
                return
            except ConnectionRefusedError:
                if time.time() - start_time > timeout:
                    self.close()
                    raise RuntimeError(f"[SimInterface] Timeout: Impossible to connect at {host}:{port} after {timeout} seconds")
                    
                print(f"[SimInterface] Socket not ready, retry in {delay} seconds...")
                time.sleep(delay)

    def read_reward_from_summary(self):

        files = glob.glob("trace*summary.csv")

        try:
            if files and os.path.exists(files[0]):

                with open(files[0], newline='') as csvfile:
                    reader = list(csv.DictReader(csvfile, delimiter=';'))

                    # No new completed jobs
                    if not reader or self.row_summary_file_status == len(reader):
                        time = 0
                        job_id = -1
                        energy = 0

                    else:
                        last_row = reader[-1]
                        job_id = last_row['job_id']
                        self.row_summary_file_status = len(reader)

                        # Simulation completed
                        if job_id == "IDLE":
                            time = float(last_row['end_time'])
                            start_time = None
                            end_time = None

                        # Intermediate step
                        else:
                            start_time = float(last_row['start_time'])
                            end_time = float(last_row['end_time'])
                            time = (end_time - start_time) * 0.001

                        # === ENERGY CALCULATION FROM TRACE ===

                        energy = 0.0

                        trace_path = files[0].replace("summary.csv", ".csv")

                        if os.path.exists(trace_path):

                            with open(trace_path, newline='') as trace_file:
                                trace_reader = csv.DictReader(
                                    trace_file, delimiter=';'
                                )

                                previous_row = next(trace_reader, None)

                                for row in trace_reader:

                                    t_previous = float(previous_row['timestamp'])
                                    t_current = float(row['timestamp'])

                                    if t_current < t_previous:
                                        raise ValueError(
                                            "Invalid trace timestamps"
                                        )

                                    # Final step: integrate the entire trace
                                    if job_id == "IDLE":
                                        dt = t_current - t_previous

                                    # Intermediate step: integrate the
                                    # interval corresponding to the last job
                                    else:
                                        dt = max(
                                            0.0,
                                            min(t_current, end_time)
                                            - max(t_previous, start_time)
                                        )

                                    active_power = float(
                                        previous_row['active_power']
                                    )
                                    idle_power = float(
                                        previous_row['idle_power']
                                    )

                                    energy += (active_power + idle_power) * dt

                                    previous_row = row

                            # Keep original reward shaping factor
                            if job_id != "IDLE":
                                energy *= 0.001

                        else:
                            print(
                                f"[SimInterface] Warning: "
                                f"trace not found: {trace_path}"
                            )

                        # =====================================

                # Reward unchanged
                reward = - 1 * time

                return reward, job_id

            else:
                print("[Python] file summary don't found. Episode to truncate.")
                return 0, -1

        except Exception as e:
            print(f"[Python] Error reading reward: {e}")
            return 0, -1

    def read_status_and_reward_from_simulation(self):
        
        # read last row of summary to have reward and last_job_id
        reward, job_id = self.read_reward_from_summary()
        # print("energy = ",metric_value, ". Job:id =", job_id)

        # In case the simulation is not terminated 
        if job_id != "IDLE":
            
            done = False
            truncated = False
            
            """
            Read a Json line from the socket connected with the simulation processes
            """
            sock_fd = self.cpp_sock.fileno()
            rlist, _, _ = select.select([sock_fd], [], [], 2.0)

            # If no data are sent
            if not rlist:
                print("[SimInterface] Timeout: no data recived from socket, even if the simulation processes still alive")
                print("[SimInterface] Forcing the end of the simulation - Done = True")
                # metric_value, job_id = self.read_reward_from_summary() #### ???
                # print("energy = ",metric_value, ". Job:id =", job_id)
                done = True
                truncated = False
                return PowercapStatus(0,0,0,0,0,0,0,[],[],reward,done,truncated)

            line = self.cpp_file_in.readline()
            # print("[Python] data received from C:", line)

            if not line.strip():
                print("[SimInterface] Socket empty, even if the simulation processes still alive")
                print("[SimInterface] Forcing the end of the simulation - Done = True")
                done = True
                truncated = False
                return PowercapStatus(0,0,0,0,0,0,0,[],[],reward,done,truncated)

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print("[SimInterface] JSON not valid, even if the simulation processes still alive")
                print("[SimInterface] Forcing the end of the simulation - Done = True")
                done = True
                truncated = False
                return PowercapStatus(0,0,0,0,0,0,0,[],[],reward,done,truncated)

            # Parsing of the Json
            total_nodes      = data.get("total_nodes", 0)
            idle_nodes       = data.get("idle_nodes", 0)
            released         = data.get("released", 0)
            requested        = data.get("requested", 0)
            total_idle_power = data.get("total_idle_power", 0)
            current_power    = data.get("current_power", 0)
            total_powercap   = data.get("total_powercap", 0)
            greedy_nodes     = data.get("greedy_nodes", [])

            greedy_data_list = [
                GreedyBytes(
                    requested=g.get("requested", 0),
                    stress=g.get("stress", 0),
                    extra_power=g.get("extra_power", 0)
                )
                for g in data.get("greedy_data", [])
            ]

            return PowercapStatus(
                total_nodes, idle_nodes, released, requested,
                total_idle_power, current_power, total_powercap,
                greedy_nodes, greedy_data_list, reward, done, truncated
            )

        # The simulatio interminted
        else:

            print("[SimInterface] Simulation Terminated with no problem")
            done = True
            truncated = False
            return PowercapStatus(0,0,0,0,0,0,0,[],[],reward,done,truncated)

    def send_action_to_cpp(self, action: dict):
        if self.cpp_file_out.closed:
            raise RuntimeError("Socket closed")

        try:
            out_line = json.dumps(action) + "\n"
            # print(">>> sending to simulation: ", out_line)
            self.cpp_file_out.write(out_line)
            self.cpp_file_out.flush()
            # print("[Python] data sent to C:")
        except BrokenPipeError:
            raise RuntimeError("[SIm Interface] Error writing on socket TCP")

    def reset(self):
        """
        First thing to do when a training starts and when a simulation is terminated and  new one starts
        """
        
        


        while True:
            print("[SimInterface] RESET: start with a new simulation..")

            try:
                
                self.kill_all_processes()                   # kill old processes ["batsim", "batsched", "eargmd", "cluster_sim"]
                self.remove_metric_files_and_save_summary() # remove all the files generated by simulation processess, and save summary file in history
                self.run_simulation()                       # run simulation processes
                self.connect_at_simulation_socket()         # connectin to the socket with the simulation
                self.row_summary_file_status = 0            # reset the number of the row to read in a sunmmary file 
                self.status = self.read_status_and_reward_from_simulation()  # read the first status of the Cluster in the simulation
                return self.status
            except Exception as e:
                print(f"Some error occurred during the RESET {e}")
                print("I'll Retry \n")

    def step(self, action: dict):
        
        """
        1 Step for evrey job submitted and 1 step for every job terminated.
        """

        # Increase the step counter
        self.step_counter += 1
        if (self.step_counter % 500 == 0):
            print("step = ", self.step_counter)
        
        # Send action
        self.send_action_to_cpp(action)

        # Read action
        return self.read_status_and_reward_from_simulation()

    def close(self):

        print("[SimInterface] Closing the training")

        try:
            if hasattr(self, 'eargmd_proc') and self.eargmd_proc and not self.eargmd_proc.stdin.closed:
                self.eargmd_proc.stdin.close()
            if hasattr(self, 'eargmd_proc') and self.eargmd_proc:
                self.eargmd_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if hasattr(self, 'eargmd_proc') and self.eargmd_proc:
                self.eargmd_proc.kill()
        finally:
            try:
                if hasattr(self, 'eargmd_proc') and self.eargmd_proc:
                    stderr_output = self.eargmd_proc.stderr.read()
                    if stderr_output:
                        print("[SimInterface] stderr dal C++:", stderr_output.strip())
            except Exception:
                pass
            # Kill everything else
            self.kill_all_processes()

            try:
                self.cpp_file_in.close() # chiudiamo le socket
                self.cpp_sock.close()
            except Exception:
                pass