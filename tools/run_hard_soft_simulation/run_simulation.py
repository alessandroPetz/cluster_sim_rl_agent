#!/usr/bin/env python3
import subprocess
import os
import time
import signal
from pathlib import Path

#################################
# PARAMETRI ESPERIMENTO DA SETTARE
num_nodes=1024
perc_cluster=95
num_jobs=2000
def_power_cap=680
#################################


file_cluster_arch = f"cluster_{num_nodes}nodes.xml"
file_power_profile = f"power_{num_jobs}jobs.txt"
file_workload = f"workload_{num_jobs}jobs_{num_nodes}nodes_{perc_cluster}.json"


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
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent.parent

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
        f"batsim -p /home/apetrella/Workspace/Barcelona/input_files/experiment1_review/{file_cluster_arch} --mmax-workload "
        f"-w /home/apetrella/Workspace/Barcelona/input_files/experiment1_review/{file_workload} -E",
        os.path.join(SCRIPT_DIR, "batsim.log"),
    ),

    # Terminal 3
    (
        "batsched -v easy_bf --verbosity=debug",
        os.path.join(SCRIPT_DIR, "batsched.log"),
    ),

    # Terminal 4
    (
        f"export CLUSTER_SIM_NUM_NODES={num_nodes} && "
        f"export CLUSTER_SIM_DEF_POWERCAP={def_power_cap} && "
        "source/ear_private/src/tools/cluster_sim " 
        f"test_tag input_files/experiment1_review/{file_power_profile}",
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