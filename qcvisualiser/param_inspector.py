from qcvisualiser.inspector import Inspector


class ParamInspector(Inspector):

    # The plot x and y axes will default to the first two elements.
    axis_map = {
        "Time": "param_time",
        "Parameter": "params",
        "Frequency": "param_freq"
    }

    _selection_parameters = (
        Inspector._selection_parameters + 
        ["antenna", "direction", "param_name"]
    )

    def __init__(self, data_path, data_field, flag_field, **params):

        super().__init__(data_path, data_field, flag_field, **params)

        # Update "param_name" to "Parameter Name" for cosmetic consistency.
        self.param.param_name.label = "Paramater Name"

    def update_selection(self, event=None):
        self.dm.set_selection(
            antenna=self.antenna,
            direction=self.direction,
            param_name=self.param_name
        )
