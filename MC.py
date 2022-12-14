import numpy as np

class MonteCarlo:
    
    """A class to simulate molecular dynamics in a monte carlo sampling fashion
    Parameters
    ---------------------------------------
    beta: 1/k_BT (in non dimensional units)
    iterations: number of monte carlo iterations
    positions: initial positions
    simulation_cell: PBC Box
    cutoff: force/energy cutoff for Lennard Jonnes interaction"""
    
    def __init__(self, cutoff, beta, simulation_cell, iterations, positions):
        self.cutoff = cutoff
        self.beta = beta
        self.simulation_cell = simulation_cell
        self.iterations = iterations
        self.init_positions = positions
        self.Ec = self.Lennard_Jones(self.cutoff)[0]
        self.fc = self.Lennard_Jones(self.cutoff)[1]

    def calculate_distance(self, r1, r2):
        return np.sqrt(np.sum((r1-r2)**2))

    def Lennard_Jones(self, r):
        r_6 = r**(-6)
        r_12 = r_6**2
        return 4*(r_12 - r_6), 4*(12*r_12/r - 6*r_6/r)

    def calculate_force_energy_pressure(self, positions):
        """Returns forces and energy from LJ and virial term of pressure"""
        N = len(positions)
        energy = 0
        forces = np.zeros((N,N,3))
        pressure = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
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
                if r > self.cutoff: #continuous energy - continuous force cutoff implementation
                    continue
                E, f = self.Lennard_Jones(r)
                u = (img-R1)/r
                forces[i, j] = (f-self.fc)*u        #Force exerted on atom j by atom i
                forces[j, i] = -(f-self.fc)*u   #Equal and opposite force exerted on atom i by atom j
                pressure[i, j] = np.dot(forces[i, j], u*r)
                energy += E - self.Ec + (r-self.cutoff)*self.fc
        #sum up the forces on atoms
        return energy, np.sum(forces, axis=0), np.sum(pressure)/3/np.prod(self.simulation_cell)

    def calculate_change_energy_pressure(self, positions, index, step):
        """Returns the change in energy and pressure after an atom at
        index is perturbed"""
        old_pos = positions[index]
        new_pos = positions[index] + step

        #Apply periodic boundary conditions
        new_pos[0] %= self.simulation_cell[0]
        new_pos[1] %= self.simulation_cell[1]
        new_pos[2] %= self.simulation_cell[2]

        pressure_old, pressure_new = 0, 0
        E_old, E_new = 0, 0
        
        for i in range(len(positions)):
            if i == index:
                continue
            neighbor = positions[i]
            #Get nearest image and interaction energy wrt to old position:
            img = np.copy(neighbor)
            if abs(old_pos[0]-neighbor[0]) > self.simulation_cell[0]/2:
                img[0] = img[0] + np.sign(old_pos[0]-neighbor[0])*self.simulation_cell[0]
            if abs(old_pos[1]-neighbor[1]) > self.simulation_cell[1]/2:
                img[1] = img[1] + np.sign(old_pos[1]-neighbor[1])*self.simulation_cell[1]
            if abs(old_pos[2]-neighbor[2]) > self.simulation_cell[2]/2:
                img[2] = img[2] + np.sign(old_pos[2]-neighbor[2])*self.simulation_cell[2]
            r = self.calculate_distance(old_pos, img)
            if r < self.cutoff: #continuous energy - continuous force cutoff implementation
                E, f = self.Lennard_Jones(r)
                E_old += E - self.Ec + (r-self.cutoff)*self.fc
                u = (img-old_pos)/r
                pressure_old += np.dot((f-self.fc)*u, u*r)

            #Get nearest image and interaction energy wrt to new position:
            img = np.copy(neighbor)
            if abs(new_pos[0]-neighbor[0]) > self.simulation_cell[0]/2:
                img[0] = img[0] + np.sign(new_pos[0]-neighbor[0])*self.simulation_cell[0]
            if abs(new_pos[1]-neighbor[1]) > self.simulation_cell[1]/2:
                img[1] = img[1] + np.sign(new_pos[1]-neighbor[1])*self.simulation_cell[1]
            if abs(new_pos[2]-neighbor[2]) > self.simulation_cell[2]/2:
                img[2] = img[2] + np.sign(new_pos[2]-neighbor[2])*self.simulation_cell[2]
            r = self.calculate_distance(new_pos, img)
            if r < self.cutoff: #continuous energy - continuous force cutoff implementation
                E, f = self.Lennard_Jones(r)
                E_new += E - self.Ec + (r-self.cutoff)*self.fc
                u = (img-new_pos)/r
                pressure_new += np.dot((f-self.fc)*u, u*r)

        delta_E = E_new - E_old
        delta_P = pressure_new - pressure_old

        return delta_E, delta_P/3/np.prod(self.simulation_cell)


    def perturbation(self, positions):
        """Select an atom at random and perturb in a random direction.
        Return the associated change in energy and pressure"""
        natoms = len(positions)
        atom_index = np.random.randint(0, natoms)
        
        #Take a step in a random direction
        phi = np.random.uniform(low=0, high=2*np.pi)
        theta = np.random.uniform(low=0, high=np.pi)
        step = np.zeros(3)
        step[0] = self.step_size*np.sin(theta)*np.cos(phi)
        step[1] = self.step_size*np.sin(theta)*np.sin(phi)
        step[2] = self.step_size*np.cos(theta)
        
        delta_E, delta_P = self.calculate_change_energy_pressure(positions, atom_index, step)
        
        #Create perturbed structure
        perturbation = np.copy(positions)
        perturbation[atom_index] += step
        perturbation[atom_index, 0] %= self.simulation_cell[0]
        perturbation[atom_index, 1] %= self.simulation_cell[1]
        perturbation[atom_index, 2] %= self.simulation_cell[2]

        return perturbation, delta_E, delta_P       

    def run(self):
        """Main function that runs monte carlo iterations"""
        E = np.zeros(self.iterations)
        P = np.zeros(self.iterations)
        curr_positions = self.init_positions # Initialize positions
        E_curr, forces, P_curr = self.calculate_force_energy_pressure(curr_positions)
        self.step_size = np.log(2)/self.beta/np.mean(np.linalg.norm(forces, axis=1))
        for i in range(self.iterations):
            new_positions, delta_E, delta_P = self.perturbation(curr_positions)
            E_new, P_new = E_curr + delta_E, P_curr + delta_P
            if delta_E < 0 or np.random.rand(1) < np.exp(-self.beta*delta_E):
                E_curr = E_new
                P_curr = P_new
                curr_positions = new_positions
            E[i] = E_curr
            P[i] = P_curr
        
        #Return energy and total pressure (ideal gas term at T + virial term from monte carlo)
        return E, P + len(self.init_positions)/self.beta/np.prod(self.simulation_cell)

def main():

    #Constants to non dimensionalize:
    ep = 1.66e-21 #Joules
    sig = 3.4e-10 #meters
    kB = 1.380649e-23 #Joule/Kelvin
    amu = 1.66054e-27 #kg
    m = 39.948*amu #kg

    positions = np.loadtxt('liquid256.txt')
    L = 6.8
    simulation_cell = [L, L, L]
    T_ = 100
    beta = 1/(T_*kB/ep)
    cutoff = 2.5
    iterations = int(10**6)
    mc = MonteCarlo(cutoff, beta, simulation_cell, iterations, positions)
    E, P = mc.run()
    np.savez('MC', E=E, P=P)

if __name__ == '__main__':
    main()