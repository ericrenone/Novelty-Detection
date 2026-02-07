import serial
import time
import random
import tkinter as tk
from tkinter import messagebox
from colorama import Fore, init

# ---------------- CONFIG ----------------

PORT = "COM6"
BAUD = 115200
ROUNDS = 1000
SEED = 42

ANOMALY_WINDOWS = [200, 500, 850]
ANOMALY_RADIUS = 15
ENERGY_THRESHOLD = 120

STEP_DELAY = 0.01

# --------------------------------------

init(autoreset=True)


def is_anomaly_round(i: int) -> bool:
    return any(abs(i - t) <= ANOMALY_RADIUS for t in ANOMALY_WINDOWS)


def run_phd_experiment():

    random.seed(SEED)

    metrics = {
        "normal": 0,
        "novel": 0,
        "total_energy": 0,
        "max_energy": 0
    }

    energy = 0

    print(Fore.CYAN + "=== STARTING WATERFALL ANOMALY EXPERIMENT ===")

    try:
        with serial.Serial(PORT, BAUD, timeout=0.05) as ser:

            ser.reset_input_buffer()
            ser.reset_output_buffer()

            start_time = time.perf_counter()

            for i in range(1, ROUNDS + 1):

                # ----- Deterministic stimulus -----
                if is_anomaly_round(i):
                    char = random.choice("!@#$%^&*")
                else:
                    char = random.choice("abcdefg ")

                # ----- Hardware transmit -----
                ser.write(char.encode("ascii"))

                # ----- Software mirror accumulator -----
                char_val = ord(char)
                energy = (energy >> 1) + char_val     # fast + bounded
                energy = min(energy, 10_000)          # overflow safety

                metrics["total_energy"] += energy
                metrics["max_energy"] = max(metrics["max_energy"], energy)

                # ----- Classification -----
                is_novel = energy >= ENERGY_THRESHOLD

                if is_novel:
                    metrics["novel"] += 1
                    color = Fore.RED
                    tag = "[ NOVEL ]"
                else:
                    metrics["normal"] += 1
                    color = Fore.GREEN
                    tag = "[ NORMAL]"

                print(
                    f"{color}{tag} "
                    f"Round {i:04d} | "
                    f"Char '{char}' | "
                    f"Energy {energy:04d}"
                )

                # ----- Precise pacing -----
                next_tick = start_time + i * STEP_DELAY
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except serial.SerialException as e:
        print(Fore.YELLOW + f"Serial error: {e}")
        return

    show_summary(metrics)


def show_summary(m):

    root = tk.Tk()
    root.withdraw()

    avg_energy = m["total_energy"] / ROUNDS
    detection_rate = (m["novel"] / ROUNDS) * 100

    summary = (
        "=== EXPERIMENT COMPLETE ===\n\n"
        f"Total Samples: {ROUNDS}\n"
        f"Normal Patterns: {m['normal']}\n"
        f"Novelty Events: {m['novel']}\n"
        f"Detection Rate: {detection_rate:.2f}%\n\n"
        f"Peak Energy: {m['max_energy']}\n"
        f"Average Energy: {avg_energy:.2f}\n\n"
        "FPGA Path: O(1) deterministic latency\n"
        "Software Mirror: causal accumulator model"
    )

    messagebox.showinfo("PhD-Grade Metrics Summary", summary)
    root.destroy()


if __name__ == "__main__":
    run_phd_experiment()
