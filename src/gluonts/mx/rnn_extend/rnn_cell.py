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
__all__ = ['RNNZoneoutCell']

# Third-party imports
from mxnet.gluon.rnn import ModifierCell, BidirectionalCell, SequentialRNNCell

class RNNZoneoutCell(ModifierCell):
    """Applies Zoneout on base cell."""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0.):
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

    def __repr__(self):
        s = '{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def _alias(self):
        return 'zoneout'

    def reset(self):
        super(RNNZoneoutCell, self).reset()
        self._prev_output = None

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
        new_states += ([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                        zip(next_states[1:], states[1:])] if p_states != 0. else next_states[1:])

        self._prev_output = output

        return output, new_states
