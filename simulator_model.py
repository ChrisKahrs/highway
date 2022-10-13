from highwaySim import highwaySim   
from typing import Dict, Any

class SimulatorModel:
    """
    Manages the simulation by implementing the reset and step methods required for a Bonsai simulator.
    """

    def __init__(self):
        """ Perform global initialization here if needed before running episodes. """
        self.highwaysim = highwaySim()

    def reset(self, config) -> Dict[str, Any]:
         # self.adder = Adder(config['initial_value'])
        
        return_dict = { 'sim_halted': False }
        return_dict.update(self.highwaysim.reset(config))
        
        return return_dict

    def step(self, action) -> Dict[str, Any]:
        """ Apply the specified action and perform one simulation step. """
        # TODO: Perform a simulation step using the values in the action dictionary.
        return_dict = { 'sim_halted': False }
        return_dict.update(self.highwaysim.step(action))
        
        return return_dict
