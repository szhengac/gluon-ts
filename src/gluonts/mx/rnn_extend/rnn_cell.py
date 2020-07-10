# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=no-member, invalid-name, protected-access, no-self-use
# pylint: disable=too-many-branches, too-many-arguments, no-self-use
# pylint: disable=too-many-lines, arguments-differ
"""Definition of various recurrent neural network cells."""
__all__ = ['RNNZoneoutCell', 'AccumulateStatesCell']

# Third-party imports
from mxnet import symbol, ndarray
from mxnet.gluon.rnn import ModifierCell, BidirectionalCell, SequentialRNNCell
from mxnet.gluon.rnn.rnn_cell import _format_sequence, _mask_sequence_variable_length, _get_begin_state
from mxnet.gluon import tensor_types

class RNNZoneoutCell(ModifierCell):
    """Applies Zoneout on base cell."""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0., preserve_raw_output=False):
        assert not isinstance(base_cell, BidirectionalCell), \
            "BidirectionalCell doesn't support zoneout since it doesn't support step. " \
            "Please add RNNZoneoutCell to the cells underneath instead."
        assert not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, \
            "Bidirectional SequentialRNNCell doesn't support zoneout. " \
            "Please add RNNZoneoutCell to the cells underneath instead."
        super(RNNZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None
        self.preserve_raw_output = preserve_raw_output

    def __repr__(self):
        s = '{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def _alias(self):
        return 'rnnzoneout'

    def reset(self):
        super(RNNZoneoutCell, self).reset()
        self._prev_output = None

    def state_info(self, batch_size=0):
        cell_state_info = self.base_cell.state_info(batch_size)
        if self.preserve_raw_output:
            # placeholder for the raw output
            cell_state_info.append(cell_state_info[0])
        return cell_state_info

    def begin_state(self, func=symbol.zeros, **kwargs):
        assert not self._modified, \
            "After applying modifier cells the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(func=func, **kwargs)
        if self.preserve_raw_output:
            # placeholder for the raw output
            begin.append(None)
        self.base_cell._modified = True
        return begin

    def hybrid_forward(self, F, inputs, states):
        cell, p_outputs, p_states = self.base_cell, self.zoneout_outputs, self.zoneout_states
        next_output, next_states = cell(inputs, states)
        mask = (lambda p, like: F.Dropout(F.ones_like(like), p=p))

        prev_output = self._prev_output
        if prev_output is None:
            prev_output = F.zeros_like(next_output)

        output_mask = mask(p_outputs, next_output)
        output = (F.where(output_mask, next_output, prev_output)
                  if p_outputs != 0. else next_output)
        
        # only for RNN, the first element of states is output
        # use the same mask as output, instead of simply copy output to the first element
        # in case that the base cell is ResidualCell
        new_states = [F.where(output_mask, next_states[0], states[0])
                          if p_outputs != 0. else next_states[0]]
        if self.preserve_raw_output:
            # the output is stored in the last 2 elements of states, thus skip them
            new_states.extend([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                               zip(next_states[1:], states[1:-1])] if p_states != 0. else next_states[1:])
            # store raw output
            new_states.append(next_states[0])
        else:
            new_states += ([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                            zip(next_states[1:], states[1:])] if p_states != 0. else next_states[1:])

        self._prev_output = output

        return output, new_states


class AccumulateStatesCell(ModifierCell):
    """
    Accumulate part of the states in base cell.
    To make it work, AccumulateStatesCell must be the last ModifierCell when stacking
    """
    def __init__(self, base_cell, index_list):
        super(AccumulateStatesCell, self).__init__(base_cell)
        self.index_list = index_list

    def hybrid_forward(self, F, inputs, states):
        # identity mapping
        next_output, next_states = self.base_cell(inputs, states)

        return next_output, next_states
    
    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        self.reset()

        inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, False)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        states = begin_state
        outputs = []
        all_states = []
        accumulated_states_list = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)
            if valid_length is not None:
                all_states.append(states)
                selected_states = []
                for index in self.index_list:
                    selected_states.append(states[index])
                accumulated_states_list.append(selected_states)
        if valid_length is not None:
            states = [F.SequenceLast(F.stack(*ele_list, axis=0),
                                     sequence_length=valid_length,
                                     use_sequence_length=True,
                                     axis=0)
                      for ele_list in zip(*all_states)]
            outputs = _mask_sequence_variable_length(F, outputs, length, valid_length, axis, True)

            # accumulate states
            # accumulated_states = [_format_sequence(length, ele_list, layout, merge_outputs)
            #                       for ele_list in zip(*accumulated_states_list)]
            accumulated_states = [list(ele_list) for ele_list in zip(*accumulated_states_list)]
            states.extend(accumulated_states)

        outputs, _, _, _ = _format_sequence(length, outputs, layout, merge_outputs)

        return outputs, states
