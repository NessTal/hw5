import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import pandas as pd


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
            axs[idx].scatter(t, stim_data.data[rat_id].isel(repetition=rep_number, electrode=elec).values, s=0.5)
            idx += 1
        plt.show()
        pass

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        experimenter_to_averages_dict = {}
        for rat in self.data.data_vars:
            m = self.data[rat].values.mean()
            experimenter = self.data[rat].attrs['Experimenter name']
            if experimenter in experimenter_to_averages_dict.keys():
                experimenter_to_averages_dict[experimenter].append(m)
            else:
                experimenter_to_averages_dict[experimenter] = [m]

        experimenters_stats = pd.DataFrame(index=experimenter_to_averages_dict.keys(), columns=('mean', 'stdev', 'median'))
        for experimenter in experimenter_to_averages_dict.keys():
            averages = experimenter_to_averages_dict[experimenter]
            if len(averages) > 1:
                experimenters_stats.loc[experimenter] = (st.mean(averages), st.stdev(averages), st.median(averages))
            else:
                experimenters_stats.loc[experimenter] = (averages[0], 0, averages[0])

        fig, axs = plt.subplots()
        axs.bar(experimenters_stats.index.values, experimenters_stats['mean'], yerr=experimenters_stats['stdev'])
        axs.plot(experimenters_stats.index.values, experimenters_stats['median'],'ro')
        plt.ylim(0.495,0.505)
        plt.show()
        return experimenters_stats


def mock_stim_data(n_rats=10) -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    rats = np.arange(n_rats)
    dims = ('time', 'electrode', 'repetition')
    coords = {'time': np.linspace(0, 2, num=10000), 'electrode': np.arange(10), 'repetition': np.arange(4)}
    rat_to_data_dict = {}
    for rat in rats:
        attrs = {'Rat ID': rat, 'Room temp': np.random.randint(10,30), 'Room humidity': np.random.randint(30,65), 'Experimenter name': np.random.choice(['A', 'B', 'C']), 'Rate gender': np.random.choice(['Male', 'Female'])}
        rat_to_data_dict[rat] = xr.DataArray(np.random.random((10000, 10, 4)), dims=dims, coords=coords, attrs=attrs)
    data = xr.Dataset(rat_to_data_dict)
    return VisualStimData(data=data)


if __name__ == '__main__':
    stim_data = mock_stim_data()
    stim_data.plot_electrode(rep_number=2, rat_id=1, elec_number=(0, 2, 5))
    stim_data.experimenter_bias()
