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
__all__ = ['ZoneoutCell', 'RNNZoneoutCell', 'AccumulateStatesCell', "ActivationRegularizationLoss", "TemporalActivationRegularizationLoss"]

# Third-party imports
from mxnet import symbol, ndarray
from mxnet.gluon.rnn import ModifierCell, BidirectionalCell, SequentialRNNCell
from mxnet.gluon.rnn.rnn_cell import _format_sequence, _mask_sequence_variable_length, _get_begin_state
from mxnet.gluon import tensor_types
from mxnet.gluon.loss import Loss

class ZoneoutCell(ModifierCell):
    """Applies Zoneout on base cell."""
    def __init__(self, base_cell, zoneout_outputs=0., zoneout_states=0., preserve_raw_output=False):
        assert not isinstance(base_cell, BidirectionalCell), \
            "BidirectionalCell doesn't support zoneout since it doesn't support step. " \
            "Please add ZoneoutCell to the cells underneath instead."
        assert not isinstance(base_cell, SequentialRNNCell) or not base_cell._bidirectional, \
            "Bidirectional SequentialRNNCell doesn't support zoneout. " \
            "Please add ZoneoutCell to the cells underneath instead."
        super(ZoneoutCell, self).__init__(base_cell)
        self.zoneout_outputs = zoneout_outputs
        self.zoneout_states = zoneout_states
        self._prev_output = None
        self.preserve_raw_output = preserve_raw_output

    def __repr__(self):
        s = '{name}(p_out={zoneout_outputs}, p_state={zoneout_states}, {base_cell})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)

    def _alias(self):
        return 'zoneout'

    def reset(self):
        super(ZoneoutCell, self).reset()
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

        output = (F.where(mask(p_outputs, next_output), next_output, prev_output)
                  if p_outputs != 0. else next_output)
        
        if self.preserve_raw_output:
            # the output is stored in the last element of states, thus skip it
            new_states = [F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                                  zip(next_states, states[0:-1])] if p_states != 0. else next_states
            # store raw output
            new_states.append(next_states[0])
        else:
            new_states = [F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                                  zip(next_states, states)] if p_states != 0. else next_states

        self._prev_output = output

        return output, new_states

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
            # the output is stored in the last element of states, thus skip it
            new_states.extend([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
                               zip(next_states[1:], states[1:-1])] if p_states != 0. else next_states[1:])
            # store raw output
            new_states.append(next_states[0])
        else:
            new_states.extend([F.where(mask(p_states, new_s), new_s, old_s) for new_s, old_s in
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

    def state_info(self, batch_size=0):
        cell_state_info = self.base_cell.state_info(batch_size)
        for index in self.index_list:
            # placeholder for the accumulated states
            cell_state_info.append({'shape': (-1), '__layout__': 'T', 'state_index': index})
        return cell_state_info

    def begin_state(self, func=symbol.zeros, **kwargs):
        assert not self._modified, \
            "After applying modifier cells the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(func=func, **kwargs)
        for index in self.index_list:
            # placeholder for the accumulated states
            begin.append([])
        self.base_cell._modified = True
        return begin

    def hybrid_forward(self, F, inputs, states):
        # identity mapping
        next_output, next_states = self.base_cell(inputs, states)
        # append accumulated states
        accumulated_states = []
        for i, index in enumerate(self.index_list):
            accumulated_states.append(states[len(states)-len(self.index_list)+i] + [next_states[index]])
        next_states.extend(accumulated_states)

        return next_output, next_states

class ActivationRegularizationLoss(Loss):
    r"""Computes Activation Regularization Loss. (alias: AR)
    The formulation is as below:
    .. math::
        L = \alpha L_2(h_t)
    where :math:`L_2(\cdot) = {||\cdot||}_2, h_t` is the output of the RNN at timestep t.
    :math:`\alpha` is scaling coefficient.
    The implementation follows the work::
        @article{merity2017revisiting,
          title={Revisiting Activation Regularization for Language RNNs},
          author={Merity, Stephen and McCann, Bryan and Socher, Richard},
          journal={arXiv preprint arXiv:1708.01009},
          year={2017}
        }
    Parameters
    ----------
    alpha : float, default 0
        The scaling coefficient of the regularization.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, alpha=0, weight=None, time_axis=0, batch_axis=1, **kwargs):
        super(ActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._alpha = alpha
        self._time_axis = time_axis
        self._batch_axis = batch_axis

    def __repr__(self):
        s = 'ActivationRegularizationLoss (alpha={alpha})'
        return s.format(alpha=self._alpha)

    def hybrid_forward(self, F, *states): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        states : list
            the stack outputs from RNN, which consists of output from each time step (TNC).
        Returns
        --------
        loss : NDArray
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        # pylint: disable=unused-argument
        if self._alpha != 0:
            if states:
                means = []
                for state in states:
                    if isinstance(state, list):
                        state = F.stack(*state, axis=self._time_axis)
                    means.append(self._alpha * state.__pow__(2).mean(axis=self._batch_axis, exclude=True))
                return F.add_n(*means)
            else:
                return F.zeros(1)
        return F.zeros(1)

class TemporalActivationRegularizationLoss(Loss):
    r"""Computes Temporal Activation Regularization Loss. (alias: TAR)
    The formulation is as below:
    .. math::
        L = \beta L_2(h_t-h_{t+1})
    where :math:`L_2(\cdot) = {||\cdot||}_2, h_t` is the output of the RNN at timestep t,
    :math:`h_{t+1}` is the output of the RNN at timestep t+1, :math:`\beta` is scaling coefficient.
    The implementation follows the work::
        @article{merity2017revisiting,
          title={Revisiting Activation Regularization for Language RNNs},
          author={Merity, Stephen and McCann, Bryan and Socher, Richard},
          journal={arXiv preprint arXiv:1708.01009},
          year={2017}
        }
    Parameters
    ----------
    beta : float, default 0
        The scaling coefficient of the regularization.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, beta=0, weight=None, time_axis=0, batch_axis=1, **kwargs):
        super(TemporalActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._beta = beta
        self._time_axis = time_axis
        self._batch_axis = batch_axis

    def __repr__(self):
        s = 'TemporalActivationRegularizationLoss (beta={beta})'
        return s.format(beta=self._beta)

    def hybrid_forward(self, F, *states): # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        states : list
            the stack outputs from RNN, which consists of output from each time step (TNC).
        Returns
        --------
        loss : NDArray
            loss tensor with shape (batch_size,). Dimensions other than batch_axis are averaged out.
        """
        # pylint: disable=unused-argument
        if self._beta != 0:
            if states:
                means = []
                for state in states:
                    if isinstance(state, list):
                        state = F.stack(*state, axis=self._time_axis)
                    sub_state_1 = F.slice_axis(state, axis=self._time_axis, begin=1, end=None)
                    sub_state_2 = F.slice_axis(state, axis=self._time_axis, begin=0, end=-1)
                    sub_state_diff = F.elemwise_sub(sub_state_1, sub_state_2)
                    means.append(self._beta * sub_state_diff.__pow__(2).mean(axis=self._batch_axis, exclude=True))
                return F.add_n(*means)
            else:
                return F.zeros(1)
        return F.zeros(1)