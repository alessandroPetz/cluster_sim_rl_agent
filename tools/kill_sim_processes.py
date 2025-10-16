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