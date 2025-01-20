from qcvisualiser.inspector import Inspector


class GainInspector(Inspector):

    # The plot x and y axes will default to the first two elements.
    axis_map = {
        "Time": "gain_time",
        "Amplitude": "amplitude",
        "Frequency": "gain_freq",
        "Phase": "phase",
        "Real": "real",
        "Imaginary": "imaginary"
    }

    _selection_parameters = (
        Inspector._selection_parameters + 
        ["antenna", "direction", "correlation"]
    )

    def __init__(self, data_path, data_field, flag_field, **params):

        super().__init__(data_path, data_field, flag_field, **params)

        # Ensure that amplitude is added to data on init.
        self.dm.set_otf_columns(amplitude=self.data_field)

    def update_selection(self, event=None):
        self.dm.set_selection(
            antenna=self.antenna,
            direction=self.direction,
            correlation=self.correlation
        )
