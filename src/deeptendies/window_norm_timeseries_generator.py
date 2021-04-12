from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np 
import json 
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class WindowNormTimeseriesGenerator(TimeseriesGenerator):
    """Extended from the keras TimeSeriesGenerator to add windowed normalization
    
    Utility class for generating batches of temporal data.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
    # Returns
        A [Sequence](/utils/#sequence) instance.
    # Examples
    ```python
    from keras.preprocessing.sequence import TimeseriesGenerator
    import numpy as np
    data = np.array([[i] for i in range(50)])
    targets = np.array([[i] for i in range(50)])
    data_gen = TimeseriesGenerator(data, targets,
                                   length=10, sampling_rate=2,
                                   batch_size=2)
    assert len(data_gen) == 20
    batch_0 = data_gen[0]
    x, y = batch_0
    assert np.array_equal(x,
                          np.array([[[0], [2], [4], [6], [8]],
                                    [[1], [3], [5], [7], [9]]]))
    assert np.array_equal(y,
                          np.array([[10], [11]]))
    ```
    """

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.target_scalers = {}
        self.scalers = {}

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        return (self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index, scale=True, min_max_scaler = True, target_idx = 3):
      """Gets next batch sample and targets

      Any columns at index < target_idx will not be scaled. Any columns at indices >= target_idx will be normalized.  
      Params: 
        index: Index to retrieve
        scale: boolean, if True, apply windowed normalization. 
        min_max_scaler: boolean, if True, use min max scaler, else, use standard scaler.
        target_idx: index of target column, and columns with index < target will not be normalized. Any columns at index > target_idx will be normalized
      """
      if self.shuffle:
          rows = np.random.randint(
              self.start_index, self.end_index + 1, size=self.batch_size)
      else:
          i = self.start_index + self.batch_size * self.stride * index
          rows = np.arange(i, min(i + self.batch_size *
                                  self.stride, self.end_index + 1), self.stride)

      samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                          for row in rows])
      targets = np.array([self.targets[row] for row in rows])

      if scale: 
        samples, targets = self.normalizer(samples, targets, index, min_max_scaler, target_idx = target_idx)            

      if self.reverse:
          return samples[:, ::-1, ...], targets
      return samples, targets

    def normalizer(self, samples, targets, idx, min_max_scaler=True, target_idx = 3):
      """helper to perform windowed min/max scaling
      
      Params: 
        idx: index of sample 
        target_idx: column index of target vector 
        min_max_scaler: If True, use min max scaler. Else, use standard. 
      Returns: 
        samples: sample array
        targets: target array
      """
      sample_idx = 0 
      scaler_list = []
      target_scaler_list = []
      while sample_idx < len(samples): 
        if min_max_scaler: 
          scaler = MinMaxScaler()
          target_scaler = MinMaxScaler()
        else: 
          scaler = StandardScaler()
          target_scaler = StandardScaler()
        target_scaler.fit(samples[sample_idx][:,target_idx].reshape(-1,1))
        samples[sample_idx][:, target_idx:] = scaler.fit_transform(samples[sample_idx][:, target_idx:])
        targets[sample_idx] = target_scaler.transform(targets[sample_idx].reshape(1, -1))
        scaler_list.append(scaler)
        target_scaler_list.append(target_scaler)
        sample_idx +=1
      self.target_scalers[idx] = target_scaler_list 
      self.scalers[idx] = scaler_list
      return samples, targets
    
    def unnormalize_target(self, target_vector, idx):
      """Restore target vector to original dollar feature space. 
      Params: 
        target_vector: target vector to restore
        idx: idx matching corresponding sample. Used to match scaler. 
      """
      target_idx = 0
      y = np.copy(target_vector)
      while target_idx < len(y): 
        scaler = self.target_scalers[idx][target_idx]
        y[target_idx] = scaler.inverse_transform(y[target_idx].reshape(1, -1))
        target_idx +=1
      return y

    def get_config(self):
        '''Returns the TimeseriesGenerator configuration as Python dictionary.
        # Returns
            A Python dictionary with the TimeseriesGenerator configuration.
        '''
        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError:
            raise TypeError('Data not JSON Serializable:', data)

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError:
            raise TypeError('Targets not JSON Serializable:', targets)

        return {
            'data': json_data,
            'targets': json_targets,
            'length': self.length,
            'sampling_rate': self.sampling_rate,
            'stride': self.stride,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'shuffle': self.shuffle,
            'reverse': self.reverse,
            'batch_size': self.batch_size
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.
        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.
        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(timeseries_generator_config, **kwargs)