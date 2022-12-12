import numpy as np

class MD:
    def __init__(self, positions, velocities, t_step, tot_time, simulation_cell, rc):
        self.init_positions = positions
        self.init_velocities = velocities
        self.simulation_cell = simulation_cell
        self.rc = rc
        self.tot_time = tot_time
        self.t_step = t_step
        self.Ec, self.Fc = self.Lennard_Jonnes(self.rc)
        self.N = len(self.init_positions)

    def write_xyz(positions, filename):
        n = len(positions[0])
        t_steps = len(positions)
        with open(filename, 'w') as f:
            for i in range(t_steps):
                f.write(f'{n}\n')
                f.write(f'"Lattice 4.0 4.0 4.0"\n')
                for j in range(n):
                    f.write('Ar %2.4f %2.4f %2.4f \n' % (positions[i,j,0], positions[i,j,1], positions[i,j,2]))
        return


    def calculate_distance(self, r1, r2):
        return np.sqrt(np.sum((r1-r2)**2))

    def Lennard_Jones(self, r):
        r_6 = r**(-6)
        r_12 = r_6**2
        return 4*(r_12 - r_6), 4*(12*r_12/r - 6*r_6/r)

    def calculate_force_energy_pressure(self, positions):
        #Returns forces and energy from LJ and virial term of pressures
        energy = 0
        forces = np.zeros((self.N,self.N,3))
        pressure = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i > j or i == j:
                    continue
                R1 = positions[i]
                R2 = positions[j]
                img = np.copy(R2)
                # Get nearest image
                if abs(R1[0]-R2[0]) > self.simulation_cell[0]/2:
                    img[0] = R2[0] +  np.sign(R1[0]-R2[0])*self.simulation_cell[0]
                if abs(R1[1]-R2[1]) > self.simulation_cell[1]/2:
                    img[1] = R2[1] + np.sign(R1[1]-R2[1])*self.simulation_cell[1]
                if abs(R1[2]-R2[2]) > self.simulation_cell[2]/2:
                    img[2] = R2[2] + np.sign(R1[2]-R2[2])*self.simulation_cell[2]
                r = self.calculate_distance(R1, img)
                if r > self.rc: #continuous energy - continuous force cutoff implementation
                    continue
                E, f = self.Lennard_Jones(r)
                u = (img-R1)/r
                forces[i, j] = (f-self.fc)*u        #Force exerted on atom i by atom j
                forces[j, i] = -(f-self.fc)*u   #Equal and opposite force exerted on atom j by atom i
                pressure[i, j] = np.dot(forces[i, j], u*r)
                energy += E - self.Ec + (r-self.rc)*self.fc
        #sum up the forces on atoms
        return energy, np.sum(forces, axis=0), np.sum(pressure)/3/np.prod(self.simulation_cell)

    def calculate_kinetic_energy(self, v):
        return np.sum(0.5*v**2)


    def update_velocity(self, v, F, t_step):
        return v + t_step*F

    def update_position(self, r, v):
        return r + self.t_step*v

    def run(self):
        n_steps = int(self.tot_time/self.t_step)
        positions = np.zeros((n_steps, self.N, 3))
        velocities = np.zeros((n_steps, self.N, 3))
        U = np.zeros((n_steps))
        KE = np.zeros((n_steps))
        T = np.zeros((n_steps))
        P = np.zeros((n_steps))
        positions[0] = self.init_positions
        velocities[0] = self.init_velocities
        u, forces, pressure = self.calculate_force_energy_pressure(positions[0])
        for i in range(n_steps):
            U[i] = u
            KE[i] = self.calculate_kinetic_energy(velocities[i])
            T[i] = 2*KE[i]/3/(self.N-1)
            P[i] = self.N*T[i]/np.prod(self.simulation_cell) + pressure #Ideal gas pressure + virial term
            if i == n_steps-1: #If simulation time exceeds total specified
                break
            vel = self.update_velocity(velocities[i], forces, 0.5*self.t_step)
            positions[i+1] = self.update_position(positions[i], vel)
            #Implement PBCs:
            positions[i+1, :, 0] = positions[i+1, :, 0] % self.simulation_cell[0]
            positions[i+1, :, 1] = positions[i+1, :, 1] % self.simulation_cell[1]
            positions[i+1, :, 2] = positions[i+1, :, 2] % self.simulation_cell[2]
            u, forces, pressure = self.calculate_force_energy_pressure(positions[i+1])
            velocities[i+1] = self.update_velocity(vel, forces, 0.5*self.t_step)
        return positions, velocities, U, KE, T, P


def main():
    R0 = np.loadtxt('liquid256.txt')
    V0 = np.zeros(np.shape(R0))
    t_step = 0.02
    tot_time = 5 #simulate for 5 units of non dimensional time
    simulation_cell = [6.8, 6.8, 6.8]
    rc = 2.5
    MD_simulator = MD(R0, V0, t_step, tot_time, simulation_cell, rc)
    positions, velocities, U, KE, T, P = MD_simulator.run()
    np.savez('Data', positions=positions, velocities=velocities, U=U, KE=KE, T=T, P=P)

if __name__ == '__main__':
    main()