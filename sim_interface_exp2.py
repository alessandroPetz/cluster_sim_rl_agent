# In this experiment we will train the agent with 4 different workload 
# and we test it with a 5th workload, never seen before


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
import random

WORKLOAD_TO_TEST = 5

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
                 num_nodes=1024,
                 max_watt_per_node=850,
                 cluster_max_power=800_000,
                 min_node_power=750,
                 host='localhost', 
                 port=12345,
                 # Mantieni un eventuale stato interno o connessione al simulatore
                 ):
        """
        - cmd: percorso (o nome) dell’eseguibile C++, es. "./sim"
        - num_nodes, max_watt_per_node, ecc. sono parametri che manteniamo
          per compatibilità, ma non vengono usati direttamente dal Python.
        """
        
        self.num_nodes = num_nodes
        self.max_watt_per_node = max_watt_per_node
        self.cluster_max_power = cluster_max_power
        self.min_node_power = min_node_power
        self.host = host
        self.port = port
        self.workload_for_training = 1
        self.workload_to_test = WORKLOAD_TO_TEST
        self.test_mode = False

        # self.kill_all_processes() # distruggo eventuali processi attivi
        # self.launch_all_processes()
        # self._connect_cpp_socket(host, port)

        self.step_counter = 0
        self.status = None
        self.row_summary_file_status = 0

    def _connect_cpp_socket(self, host='localhost', port=12345, timeout=5, delay=0.5):
        """
        Tenta di connettersi ripetutamente alla socket finché non è pronta,
        o finché non scade il timeout (in secondi).
        """
        print("MI connetto all socket..")
        start_time = time.time()
        while True:
            try:
                self.cpp_sock = socket.create_connection((host, port))
                self.cpp_file_in = self.cpp_sock.makefile('r')
                self.cpp_file_out = self.cpp_sock.makefile('w')
                print(f"[SimInterface] Connesso al socket {host}:{port}")
                return
            except ConnectionRefusedError:
                if time.time() - start_time > timeout:
                    self.close()
                    raise RuntimeError(f"Timeout: impossibile connettersi a {host}:{port} dopo {timeout} secondi")
                    
                # print(f"[SimInterface] Socket non pronta, ritento tra {delay} secondi...")
                time.sleep(delay)

    def launch_all_processes(self):

        

        if self.test_mode == True:
            self.workload_for_training = self.workload_to_test
        else:
            self.workload_for_training = random.choice([n for n in range(1, 6) if n != self.workload_to_test])

        print("=====================================================================")
        print("self.test_mode = ",self.test_mode)
        print("self.workload_to_test = ",self.workload_to_test)
        print(f"Leggo il file: leggo 1024_nodes_v{self.workload_for_training}.json")
        print("=====================================================================")

        # Lancio il processo C++ in lettura/scrittura su pipe
        workspace = "/home/apetrella/Workspace/Barcelona"

        #   TODO Lanciare un numero random tra 1 e 5, 
        #   finche non diverso da quello che usiamo per il test.
        os.makedirs(f"{workspace}/tmp", exist_ok=True)

        # Variabili d'ambiente
        env_eargmd = os.environ.copy()
        env_eargmd["EAR_ETC"] = f"{workspace}/EAR/etc"

        env_cluster_sim = os.environ.copy()
        env_cluster_sim["CLUSTER_SIM_NUM_NODES"] = "1024"
        env_cluster_sim["CLUSTER_SIM_DEF_POWERCAP"] = "750"

        print("[SimINterface] Avvio i processi...")
        self.eargmd_proc = subprocess.Popen(
            [f"{workspace}/source/ear_private/src/global_manager/eargmd"],
            stdout=open(f"{workspace}/tmp/eargmd.log", "w"),
            stderr=subprocess.STDOUT,
            env=env_eargmd
        )
        
        time.sleep(1)  # 🔁 Attendi un po' per sicurezza (puoi mettere anche un controllo log)

        #print("[]>>> Avvio batsim...")
        self.batsim_proc = subprocess.Popen(
            [
                "batsim",
                "-p", f"{workspace}/input_files/experiment2/cluster_energy_1024.xml",
                "--mmax-workload",
                "-w", f"{workspace}/input_files/experiment2/1024_nodes_v{self.workload_for_training}.json",
                "-E"
            ],
            stdout=open(f"{workspace}/tmp/batsim.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1)

        #print(">>> Avvio batsched...")
        self.batsched_proc = subprocess.Popen(
            ["batsched", "-v", "easy_bf", "--verbosity=debug"],
            stdout=open(f"{workspace}/tmp/batsched.log", "w"),
            stderr=subprocess.STDOUT
        )
        time.sleep(1)

        #print(">>> Avvio cluster_sim...")
        self.cluster_sim_proc = subprocess.Popen(
            [
                f"{workspace}/source/ear_private/src/tools/cluster_sim",
                "test_tag",
                f"../input_files/experiment2/cpu_1k_powerusage_v{self.workload_for_training}.txt"   
            ],
            env=env_cluster_sim,
            stdout=open(f"{workspace}/tmp/cluster_sim.log", "w"),
            stderr=subprocess.STDOUT
        )

        print("[SimInterface] Tutti i processi sono stati avviati.")

    def kill_all_processes(self):
        print("[SimInterface] Terminazione dei processi...")

        for proc, name in [
            (getattr(self, "eargmd_proc", None), "eargmd"),
            (getattr(self, "batsim_proc", None), "batsim"),
            (getattr(self, "batsched_proc", None), "batsched"),
            (getattr(self, "cluster_sim_proc", None), "cluster_sim"),
            (getattr(self, "proc", None), "eargmd (agent)")
        ]:
            if proc and proc.poll() is None:  # Se è ancora attivo
                try:
                    print(f"[SimInterface] Killing {name} (pid={proc.pid})")
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"[SimInterface] Kill forzato di {name} (pid={proc.pid})")
                    proc.kill()
                except Exception as e:
                    print(f"[SimInterface] Errore durante kill di {name}: {e}")

    def kill_all_named_processes(self,names):
        print("[SimInterface] Terminazione di processi esterni...")

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for name in names:
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

    def read_metric_file(self):


        # TODO Leggere il file
        # inizio con righe = 0
        # se sono piu righe di quelle che avevo gia letto, sommo tutto  e aggiorno valore
        # se sono uguali, ritorno 0
        # arrivato a ultima riga = IDLE, fininsco il file i righe = 0 da capo

        # la metrica che viene letta non è detto che sia giusta,
        # siamo sicuri che venga aggiornato il file in tempo????

        # me la faccio passare dal eargmd?? MOLTO MEGLIO

        # read the summry file and retrun the last energy metric *-1
        
        #time.sleep(0.2)
        files = glob.glob("trace*summary.csv")

        try:
            if files and os.path.exists(files[0]):
                # print("[Python] leggo metriche in", files[0])
                with open(files[0], newline='') as csvfile:
                    reader = list(csv.DictReader(csvfile, delimiter=';'))
                    # se il file non ha righe o abbiamo gia letto l'ultima riga
                    if not reader or self.row_summary_file_status == len(reader):
                        energy = 0
                        start_time = 0
                        end_time = 0
                        job_id = -1
                    else:
                            
                        # ci sono nuovi valori da passare come metrica
                        # per ora passo solo l'ultimo.... 
                        # TODO passare tutti quelli non letti
                        energy = 0
                        diff = len(reader) - self.row_summary_file_status
                        for i in range(1, diff + 1):
                            row = reader[-i]
                            energy = energy + float(row['energy'])
                        last_row = reader[-1]
                        #energy = float(last_row['energy'])
                        #start_time = int(last_row['start_time'])
                        #end_time = int(last_row['end_time'])
                        job_id = last_row['job_id']
                        self.row_summary_file_status = len(reader)


                #print("watts =", e, ", sec =", s)
                energy = -1 * energy
                # print("energy = ", energy)
                return energy, job_id
            else:
                print("[Python] file summary non trovato. Episodio troncato.")
                return 0,0
        except:
            print("[Python] file summary non trovato. Episodio troncato.")
            return 0,0


    def process_csv_and_save_results(self, file_path):
        
        #print(file_path)
        energy_sum = 0.0
        last_end_time = None
        last_job_id = -1
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile,delimiter=';')
            for row in reader:
                #print("Riga letta:", row)
                # Somma energy
                try:
                    energy_sum += float(row['energy'].strip())
                except (KeyError, ValueError):
                    pass

                # Ultimo end_time
                try:
                    last_end_time = float(row['end_time'].strip())
                except (KeyError, ValueError):
                    pass

                # Ultimo job_id
                try:
                    last_job_id = row['job_id'].strip()
                except (KeyError, ValueError):
                    # qualcosa si è incagliato
                    # a volte viene generato e non ha neanche una riga. quindi va in errore.
                    # eliminiamo metric files
                    # poi la simulazione ripartitrà
                    self._remove_metric_files()
                    last_job_id = -1
        
        # solo se era terminata la simulazione, altrimenti non lo salvo
        # print(last_job_id)
        if  last_job_id == "IDLE" :
            
            print(f"Somma energy: {energy_sum}")
            print(f"Ultimo end_time: {last_end_time}")
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # salvo su file i risultati
            
            # Percorso del file di output
            if self.test_mode == False:
                output_path = "history_summaries/results_training.csv"
            else:
                output_path = "history_summaries/results.csv"
                
            # Intestazioni
            header = ['timestamp', 'filename', 'energy', 'end_time','workload_n']
            nuova_riga = [timestamp, file_path, energy_sum, last_end_time, self.workload_for_training]
            


            # Crea la cartella se non esiste
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Controlla se il file esiste già
            file_esiste = os.path.exists(output_path)

            # Scrive su file
            with open(output_path, 'a', newline='') as file:
                writer = csv.writer(file)

                # Se il file non esisteva, scrive l'intestazione
                if not file_esiste:
                    writer.writerow(header)

                # Scrive i dati
                writer.writerow(nuova_riga)

            print("Aggiunta riga: ", nuova_riga)
            print("In PATH: ", output_path)

            # CHJECK aggiungo reset 
            # nel caso abbia registrato i numeri di un test, al giro dopo siamo sicuramnete in train di nuovo
            if self.test_mode == True:
                self.test_mode = False


    def _remove_metric_files(self):

        # sposto il file summary in history, ed elimino gli altri file

        # Percorso corrente
        current_dir = os.getcwd()

        # Cartella di destinazione
        history_dir = os.path.join(current_dir, 'history_summaries')

        # Crea la cartella 'history' se non esiste
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        # Cerca file che contengono 'summary' nel nome
        for filename in os.listdir(current_dir):
            file_path = os.path.join(current_dir, filename)
            if os.path.isfile(file_path) and 'summary' in filename.lower():

                self.process_csv_and_save_results(file_path)

                # Sposta il file nella cartella 'history'
                dest_path = os.path.join(history_dir, filename)
                shutil.move(file_path, dest_path)
                print(f"Spostato: {filename} → {history_dir}")
        
        
        # ELIMINO I FILE CSV e TRACE in più

        files = (glob.glob("*.csv"))
        try:
            for f in files:
                if f and os.path.exists(f):
                    os.remove(f)
        except:
            print("no .csv to remove")
        try:
            files = (glob.glob("*.trace"))
            for f in files:
                if f and os.path.exists(f):
                    os.remove(f)
        except:
            print("no .trace tu remove")

    def _read_status_from_cpp(self):
        
        # leggo il foglio delle metriche, se
        # se job ID != IDLE
        #         leggo metrica, done = false
        # se job ID == IDLE
        #         leggo ultima riga, done = true
        
        
         
        metric_value, job_id = self.read_metric_file()
        # print("energy = ",metric_value, ". Job:id =", job_id)
         

        # simulazione appena iniziata o non ancora terminata
        # in realtà questa cosa sembra non succedere, perchè ad un certo punt genera 2 righe
        # quella di fine job e quella di fine simulazione (IDLE)
        # allora gliela faccio leggere nel caso non si ricevono più dati
        if job_id != "IDLE":
            
            # print("processo attivo")
            done = False
            truncated = False
            
            """
            Legge una riga JSON dalla socket C++ in modo non bloccante.
            """
            sock_fd = self.cpp_sock.fileno()
            rlist, _, _ = select.select([sock_fd], [], [], 2.0)

            if not rlist:
                print("[Python] Timeout: nessun dato ricevuto, anche se il processo è ancora vivo")
                print("simulazione terminata")
                metric_value, job_id = self.read_metric_file()
                print("energy = ",metric_value, ". Job:id =", job_id)
                done = True
                truncated = False
                return PowercapStatus(0,0,0,0,0,0,0,[],[],metric_value,done,truncated)

            line = self.cpp_file_in.readline()
            # print("[Python] data received from C:", line)

            if not line.strip():
                print("[Python] Socket vuota, anceh se il processo è ancora vivo")
                # return self._check_for_final_metric_file()

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"SimInterface: JSON non valido in ingresso dal C++: {e}")

            # parsing come prima ...
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
                greedy_nodes, greedy_data_list, metric_value, done, truncated
            )


        else:

            print("simulazione terminata")
            done = True
            truncated = False
            return PowercapStatus(0,0,0,0,0,0,0,[],[],metric_value,done,truncated)

    def _send_action_to_cpp(self, action: dict):
        if self.cpp_file_out.closed:
            raise RuntimeError("Socket chiusa")

        try:
            out_line = json.dumps(action) + "\n"
            # print(">>> INVIO verso C: ", out_line)
            self.cpp_file_out.write(out_line)
            self.cpp_file_out.flush()
            # print("[Python] data sent to C:")
        except BrokenPipeError:
            raise RuntimeError("Errore di scrittura su socket TCP")

    # def reset(self):


    #     """
    #     Chiamiamo il C++ e leggiamo il primo status JSON.
    #     Non inviamo alcuna azione perché il C++, di solito, manda subito
    #     il primo powercap_status non appena parte.
    #     """
    #     # self.step_counter = 0
    #     # LEGGO il primo JSON dal C++

    #     while True:
    #         try:
    #             print("Dentro RESET")
    #             self.kill_all_processes()                   # elimino vecchi processi
    #             # self.kill_all_named_processes(["batsim", "batsched", "eargmd", "cluster_sim"]) 
    #             print("rimuovo file e salvo")
    #             self._remove_metric_files()                 # rimuovo i file generati dai processi
    #             self.row_summary_file_status = 0            # torna a 0
    #             self.launch_all_processes()                 # lancio i processi
    #             self._connect_cpp_socket()                  # mi connetto alla socket
    #             self.status = self._read_status_from_cpp()  # leggo il primo status
    #             return self.status
    #         except Exception as e:
    #             print(f"Errore durante la connessione o lettura: {e}")
    #             print("Riprovo da capo...\n")

    def reset(self, test_mode=False, workload_to_test=WORKLOAD_TO_TEST):
        """
        Reset della simulazione.
        test: bool → True se siamo in modalità test
        workload_to_test: int → quale workload usare per la fase di test
        """
        self.test_mode = test_mode
        self.workload_to_test = workload_to_test

        while True:
            try:
                print("[SimInterface] RESET iniziato. test_mode =", self.test_mode, ", workload_to_test =", self.workload_to_test)

                # Termina eventuali processi precedenti
                self.kill_all_processes()
                self._remove_metric_files()         # and save reuslts
                self.row_summary_file_status = 0

                # Avvia nuovi processi e passa i parametri
                self.launch_all_processes()

                # Connessione alla socket C++
                self._connect_cpp_socket()

                # Legge il primo status dal simulatore
                self.status = self._read_status_from_cpp()
                return self.status

            except Exception as e:
                print(f"[SimInterface] Errore durante reset: {e}. Riprovo...")
                time.sleep(1)

    def step(self, action, test_mode=False, workload_to_test=WORKLOAD_TO_TEST):
        """
        Esegue un passo di simulazione.
        action: dict → azione da inviare al simulatore
        test: bool → modalità test
        workload_to_test: int → workload da usare
        """
        self.test_mode = test_mode
        self.workload_to_test = workload_to_test

        # Invia l'azione al simulatore C++
        self._send_action_to_cpp(action)

        # Legge lo stato aggiornato dal simulatore
        self.status = self._read_status_from_cpp()
        return self.status

    def close(self):
        try:
            if hasattr(self, 'eargmd_proc') and self.eargmd_proc:
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
            self.kill_all_named_processes(["batsim", "batsched", "eargmd", "cluster_sim"])

            try:
                self.cpp_file_in.close() # chiudiamo le socket
                self.cpp_sock.close()
            except Exception:
                pass
