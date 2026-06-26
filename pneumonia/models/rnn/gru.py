from pneumonia.models.rnn.base import BaseRNNModel
from pneumonia.models.rnn.config import RNN_DEFAULT_PARAMS


class GRUModel(BaseRNNModel):
    """
    Stacked GRU forecaster. See BaseRNNModel for full documentation.

    GRU uses simpler gating (no cell state) vs LSTM — typically trains faster
    with similar accuracy on short-to-medium seasonal series.
    """

    def __init__(self, department: str, age_group: str, **kwargs):
        super().__init__(name="GRU", department=department, age_group=age_group, **kwargs)

    def _build_network(self, input_shape: tuple):
        from tensorflow.keras.layers import GRU
        return self._stack_rnn(GRU, input_shape)
