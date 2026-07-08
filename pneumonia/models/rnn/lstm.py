from pneumonia.models.rnn.base import BaseRNNModel
from pneumonia.models.rnn.config import RNN_DEFAULT_PARAMS


class LSTMModel(BaseRNNModel):
    """Stacked LSTM forecaster. See BaseRNNModel for full documentation."""

    def __init__(self, department: str, age_group: str, **kwargs):
        super().__init__(name="LSTM", department=department, age_group=age_group, **kwargs)

    def _build_network(self, input_shape: tuple):
        from tensorflow.keras.layers import LSTM
        return self._stack_rnn(LSTM, input_shape)
