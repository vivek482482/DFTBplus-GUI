import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("transmission.dat")

E = data[:,0]
T = data[:,1]

plt.figure(figsize=(7,4))
plt.semilogy(E, T)

plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission spectrum")

plt.axvline(0,color='k',linestyle='--',label="Fermi level")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
