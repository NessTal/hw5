import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class VisualStimData:
    """
    Data and methods for the visual stimulus ePhys experiment.
    The data table itself is held in self.data, an `xarray` object.
    Inputs:
        data: xr.DataArray
    Methods:
         plot_electrode
         experimenter_bias
    """
    def __init__(self, data):
        self.data = data

    def plot_electrode(self, rep_number: int, rat_id: int, elec_number: tuple=(0,)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.
        """
        fig, axs = plt.subplots(len(elec_number))
        t = stim_data.data.coords['time'].values
        idx = 0
        for elec in elec_number:
            axs[idx].scatter(t, stim_data.data.isel(repetition = rep_number, electrode = elec).values, s = 0.5)
            idx += 1
        plt.show()

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        pass


def mock_stim_data() -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    dims = ('time', 'electrode', 'repetition')
    coords = {'time': np.linspace(0,2, num = 10000), 'electrode': np.arange(10) , 'repetition': np.arange(4)}
    attrs = {'Rat ID':1, 'Room temp':25, 'Room humidity':40, 'Experimenter name': 'A', 'Rate gender':'male'}
    data = xr.DataArray(np.random.random((10000,10,4)),dims = dims, coords = coords, attrs = attrs)
    return VisualStimData(data = data)

if __name__ == '__main__':
    stim_data = mock_stim_data()
    stim_data.plot_electrode(rep_number = 2, rat_id = 1, elec_number = (0,2,5))
    stim_data.experimenter_bias()
