import numpy as np

from simulation import GraspSimulation


def main():
    simulation = GraspSimulation()
    simulation.update(np.zeros(21))
    simulation.close()


if __name__ == "__main__":
    main()
