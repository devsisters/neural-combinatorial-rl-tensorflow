import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.util import nest

LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell
LSTMStateTuple = rnn.LSTMStateTuple

def decoder_rnn(cell, inputs, encoder_outputs, sequence_length=None,
                initial_state=None, dtype=None, parallel_iterations=None,
                swap_memory=False, time_major=False, scope=None):
  if not isinstance(cell, rnn_cell_impl._RNNCell):
    raise TypeError("cell must be an instance of RNNCell")

  # For internal calculations, we transpose to [time, batch, depth]
  flat_input = nest.flatten(inputs)

  if not time_major:
    # (B,T,D) => (T,B,D)
    flat_input = tuple(array_ops.transpose(input_, [1, 0, 2])
                       for input_ in flat_input)

  parallel_iterations = parallel_iterations or 32
  if sequence_length is not None:
    sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length.get_shape().ndims not in (None, 1):
      raise ValueError(
          "sequence_length must be a vector of length batch_size, "
          "but saw shape: %s" % sequence_length.get_shape())
    sequence_length = tf.identity(  # Just to find it in the graph.
        sequence_length, name="sequence_length")

  with vs.variable_scope(scope or "decoder_rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)
    input_shape = tuple(array_ops.shape(input_) for input_ in flat_input)
    batch_size = input_shape[0][1]

    for input_ in input_shape:
      if input_[1].get_shape() != batch_size.get_shape():
        raise ValueError("All inputs should have the same batch size")

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        encoder_outputs.
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      flat_output = nest.flatten(outputs)
      flat_output = [array_ops.transpose(output, [1, 0, 2])
                     for output in flat_output]
      outputs = nest.pack_sequence_as(
          structure=outputs, flat_sequence=flat_output)

    return (outputs, final_state)

def _dynamic_rnn_loop(cell,
                      inputs,
                      encoder_outputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = input_shape[1]

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _state_size_with_prefix(size, prefix=[batch_size])
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, dtype):
    return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        tensor_array_name=base_name + name)

  output_ta = tuple(_create_ta("output_%d" % i,
                               _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
  input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                   for i in range(len(flat_input)))

  input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

  def _time_step(time, output_ta_t, state):
    """Take a time step of the dynamic RNN.
    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.
    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    input_t = tuple(ta.read(time) for ta in input_ta)
    # Restore some shape information
    for input_, shape in zip(input_t, inputs_got_shape):
      input_.set_shape(shape[1:])

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    call_cell = lambda: cell(input_t, state)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Pack state if using state tuples
    output = nest.flatten(output)

    output_ta_t = tuple(
        ta.write(time, out) for ta, out in zip(output_ta_t, output))

    return (time + 1, output_ta_t, new_state)

  _, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  final_outputs = tuple(ta.stack() for ta in output_final_ta)

  # Restore some shape information
  for output, output_size in zip(final_outputs, flat_output_size):
    shape = _state_size_with_prefix(
        output_size, prefix=[const_time_steps, const_batch_size])
    output.set_shape(shape)

  final_outputs = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)

  return (final_outputs, final_state)

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
  else:
    flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

  names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]

  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1] + [size]
    initial_state_variable = tf.get_variable(
        name, shape=shape_with_batch_dim, initializer=init())

    tile_dims = [batch_size] + [1]
    tiled_states.append(
        tf.tile(initial_state_variable, tile_dims, name=(name + "_tiled")))

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)
