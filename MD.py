#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

def calculate_distance(r1, r2):
    return np.sqrt(np.sum((r1-r2)**2))

def Lennard_Jones(r):
    r_12 = r**(-12)
    r_6 = r**(-6)
    return 4*(r_12 - r_6), 4*(12*r_12/r - 6*r_6/r)

def calculate_force_energy(positions):
    N = len(positions)
    energy = 0
    forces = np.zeros((N,N,3))
    for i in range(N):
        for j in range(N):
            if i > j or i == j:
                continue
            R1 = positions[i]
            R2 = positions[j]
            r = calculate_distance(R1, R2)
            E, f = Lennard_Jones(r) 
            u = (R2-R1)/r
            forces[i,j] = f*u        #Force exerted on atom i by atom j
            forces[j,i] = -f*u       #Equal and opposite force exerted on atom j by atom i
            energy += E
    return energy, np.sum(forces, axis=0) #sum up the forces on atoms

def calculate_kinetic_energy(v):
    return np.sum(0.5*v**2)

def update_velocity(v, F, t_step):
    return v + t_step*F

def update_position(r, v, t_step):
    return r + t_step*v

def verlet(R0, V0, t_step, tot_time):   
    n_atoms = len(R0)
    n_steps = int(tot_time/t_step)
    positions = np.zeros((n_steps, n_atoms, 3))
    velocities = np.zeros((n_steps, n_atoms, 3))
    U = np.zeros((n_steps))
    KE = np.zeros((n_steps))
    positions[0] = R0
    velocities[0] = V0
    u, forces = calculate_force_energy(positions[0])
    for i in range(n_steps):
        U[i] = u
        KE[i] = calculate_kinetic_energy(velocities[i])
        if i == int(tot_time/t_step)-1: #If simulation time exceeds total specified
            break
        vel = update_velocity(velocities[i], forces, 0.5*t_step)
        positions[i+1] = update_position(positions[i], vel, t_step)
        u, forces = calculate_force_energy(positions[i+1])
        velocities[i+1] = update_velocity(vel, forces, 0.5*t_step)
    return positions, velocities, U, KE


# In[7]:


R0 = np.loadtxt('10.txt')
V0 = np.zeros(np.shape(R0))
t_step = 0.002
tot_time = 5 #simulate for 5 units of non dimensional time
positions, velocities, U, KE = verlet(R0, V0, t_step, tot_time)




