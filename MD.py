#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import numba as nb

@nb.njit
def calculate_distance(r1, r2):
    return np.sqrt(np.sum((r1-r2)**2))

def Lennard_Jones(r):
    r_6 = r**(-6)
    r_12 = r_6**2
    return 4*(r_12 - r_6), 4*(12*r_12/r - 6*r_6/r)

def calculate_force_energy_pressure(positions, simulation_cell, rc):
    #Returns forces and energy from LJ and virial term of pressure
    N = len(positions)
    energy = 0
    forces = np.zeros((N,N,3))
    pressure = np.zeros((N,N))
    Ec, fc = Lennard_Jones(rc)
    inter_distances = []
    for i in range(N):
        for j in range(N):
            if i > j or i == j:
                continue
            R1 = positions[i]
            R2 = positions[j]
            img = np.copy(R2)
            # Get nearest image
            if abs(R1[0]-R2[0]) > simulation_cell[0]/2:
                img[0] = R2[0] +  np.sign(R1[0]-R2[0])*simulation_cell[0]
            if abs(R1[1]-R2[1]) > simulation_cell[1]/2:
                img[1] = R2[1] + np.sign(R1[1]-R2[1])*simulation_cell[1]
            if abs(R1[2]-R2[2]) > simulation_cell[2]/2:
                img[2] = R2[2] + np.sign(R1[2]-R2[2])*simulation_cell[2]
            r = calculate_distance(R1, img)
            if r > rc: #continuous energy - continuous force cutoff implementation
                continue
            E, f = Lennard_Jones(r)
            u = (img-R1)/r
            forces[i, j] = (f-fc)*u        #Force exerted on atom i by atom j
            forces[j, i] = -(f-fc)*u   #Equal and opposite force exerted on atom j by atom i
            pressure[i, j] = np.dot(forces[i, j], u)
            energy += E - Ec + (r-rc)*fc
    #sum up the forces on atoms
    return energy, np.sum(forces, axis=0), np.sum(pressure)/3/np.prod(simulation_cell)

def calculate_kinetic_energy(v):
    return np.sum(0.5*v**2)


def update_velocity(v, F, t_step):
    return v + t_step*F

def update_position(r, v, t_step):
    return r + t_step*v

def verlet(R0, V0, simulation_cell, cutoff, t_step, tot_time):   
    n_atoms = len(R0)
    n_steps = int(tot_time/t_step)
    positions = np.zeros((n_steps, n_atoms, 3))
    velocities = np.zeros((n_steps, n_atoms, 3))
    U = np.zeros((n_steps))
    KE = np.zeros((n_steps))
    T = np.zeros((n_steps))
    P = np.zeros((n_steps))
    positions[0] = R0
    velocities[0] = V0
    u, forces, pressure = calculate_force_energy_pressure(positions[0], simulation_cell, cutoff)
    for i in range(n_steps):
        U[i] = u
        KE[i] = calculate_kinetic_energy(velocities[i])
        T[i] = 2*KE[i]/3/(n_atoms-1)
        P[i] = n_atoms*T[i]/np.prod(simulation_cell) + pressure #Ideal gas pressure + virial term
        if i == int(tot_time/t_step)-1: #If simulation time exceeds total specified
            break
        vel = update_velocity(velocities[i], forces, 0.5*t_step)
        positions[i+1] = update_position(positions[i], vel, t_step)
        #Implement PBCs:
        positions[i+1, :, 0] = positions[i+1, :, 0] % simulation_cell[0]
        positions[i+1, :, 1] = positions[i+1, :, 1] % simulation_cell[1]
        positions[i+1, :, 2] = positions[i+1, :, 2] % simulation_cell[2]
        u, forces, pressure = calculate_force_energy_pressure(positions[i+1], simulation_cell, cutoff)
        velocities[i+1] = update_velocity(vel, forces, 0.5*t_step)
    return positions, velocities, U, KE, T, P


# In[7]:


R0 = np.loadtxt('liquid256.txt')
V0 = np.zeros(np.shape(R0))
t_step = 0.002
tot_time = 5 #simulate for 5 units of non dimensional time
positions, velocities, U, KE = verlet(R0, V0, t_step, tot_time)




