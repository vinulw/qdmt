import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 32

midblue = "#58C4DD"
midred = "#FF8080"
darkblue = "#236B8E"
darkred = "#CF5044"
midgreen = "#A6CF8C"
darkgreen = "#699C52"

def plot_fidelity_density(dataPath):
    dataFile = dataPath / "fidelity_density_itebd_data.csv"
    data = np.loadtxt(dataFile, delimiter=',').T
    plt.figure(figsize=(12, 12))
    plt.plot(data[0], data[1], '.', markersize=12, color=darkblue)
    plt.xlabel('Time')
    plt.ylabel('Fidelity Density with iTEBD')
    plt.grid()
    figPath = dataPath / 'fidelity_itebd.png'
    plt.savefig(figPath)
    plt.show()

def plot_trace_dist(dataPath):
    dataFile = dataPath / "local_density_itebd_data.csv"
    data = np.loadtxt(dataFile, delimiter=',').T
    plt.figure(figsize=(12, 12))
    plt.plot(data[0], data[1], '.', markersize=10, color=darkgreen)
    plt.xlabel('Time')
    plt.ylabel('Local Trace Distance with iTEBD')
    plt.grid()
    figPath = dataPath / 'trace_dist_itebd.png'
    plt.savefig(figPath)
    plt.show()

if __name__=="__main__":
    dataPath = Path("./data/16072024-171900/")
    plot_trace_dist(dataPath)
    plot_fidelity_density(dataPath)
