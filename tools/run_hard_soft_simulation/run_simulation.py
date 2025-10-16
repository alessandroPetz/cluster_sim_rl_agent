#!/usr/bin/env python3
import subprocess
import os
import time
import signal

def kill_processes_by_name(name: str):
    try:
        # Trova i PID con pgrep
        pids = subprocess.check_output(["pgrep", "-x", name], text=True).split()
    except subprocess.CalledProcessError:
        print(f"Nessun processo trovato con nome '{name}'.")
        return

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Processo {name} con PID {pid} terminato.")
        except Exception as e:
            print(f"Errore nel terminare PID {pid}: {e}")


print("RIchiudo tutti i processi in ogni caso")
kill_processes_by_name("eargmd")
kill_processes_by_name("batsim")
kill_processes_by_name("batsched")
kill_processes_by_name("cluster_sim")


# Percorso base del progetto
BASE = "/home/apetrella/Workspace/Barcelona"
# Cartella dove si trova questo script (così i log finiscono qui)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ##############
# Ho settato cwd = BASE
####################
commands = [
    # Terminal 1
    (
        f"export EAR_ETC={BASE}/EAR/etc && "
        "source/ear_private/src/global_manager/eargmd",
        os.path.join(SCRIPT_DIR, "eargmd.log"),
    ),

    # Terminal 2
    (
        "batsim -p input_files/experiment1/cluster_energy_1024.xml --mmax-workload "
        "-w input_files/experiment1/1024_nodes.json -E",
        os.path.join(SCRIPT_DIR, "batsim.log"),
    ),

    # Terminal 3
    (
        "batsched -v easy_bf --verbosity=debug",
        os.path.join(SCRIPT_DIR, "batsched.log"),
    ),

    # Terminal 4
    (
        "export CLUSTER_SIM_NUM_NODES=1024 && "
        "export CLUSTER_SIM_DEF_POWERCAP=750 && "
        "source/ear_private/src/tools/cluster_sim "
        "test_tag input_files/experiment1/cpu_1k_powerusage.txt",
        os.path.join(SCRIPT_DIR, "cluster_sim.log"),
    ),
]


procs = []
for i, (cmd, logfile) in enumerate(commands, start=1):
    log = open(logfile, "w")
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid, cwd=BASE)
    procs.append((p, log))
    print(f"[{i}/{len(commands)}] Avviato: {cmd.split()[0]} (log: {logfile})")
    time.sleep(1)  # pausa di 1 secondo tra un processo e l'altro

print("\nTutti i processi sono stati lanciati. Log salvati in:")
for _, logfile in commands:
    print(f" - {logfile}")

# cluster_sim è l'ultimo nella lista
cluster_sim_proc, cluster_sim_log = procs[-1]

try:
    cluster_sim_proc.wait()  # aspetta che cluster_sim finisca
finally:
    print("\n⚠️ cluster_sim terminato, chiudo tutti i processi...")
    for p, log in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # termina il gruppo di processi
        except ProcessLookupError:
            pass
        log.close()

print("RIchiudo tutti i processi in ogni caso")
kill_processes_by_name("eargmd")
kill_processes_by_name("batsim")
kill_processes_by_name("batsched")
kill_processes_by_name("cluster_sim")