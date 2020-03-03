# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for category learning trial sequences.

Classes:
    TrialSequence: Abstract class for a trial sequence.
    StimulusSequence: TODO
    BehaviorSequence: TODO
    AFCSequence: TODO

Functions:
    stack: Stack sequences.
    pad_trials: Pad trials.

Todo:
    * change `is_feedback` to `feedback` which allows for more than two
    levels (yes/no).
    * `kind` as a catch all to be used/abused in what ever way the user wants
    * implement `load_sequence`
    * test: group_id is composed of integers
    * test: stack of stimulus seq
    * test: stack of behavior seq
    * MAYBE `kind` shape=(n_sequence, n_trial, n_level)
    * MAYBE `group_id` shape=(n_sequence, n_trial, n_level)
    * MAYBE Allow for list of lists in response times?

"""

from abc import ABCMeta, abstractmethod
import copy
import os
from pathlib import Path

import h5py
import numpy as np


class TrialSequence(object):
    """Abstract base class.

    Attributes:
        TODO

    Methods:
        save: Save the sequence.

    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize."""
        super().__init__()
        self._version = '0.1.0'


class StimulusSequence(TrialSequence):
    """Class for storing a stimulus sequence."""

    def __init__(
            self, stimulus_id, class_id, is_feedback=None, kind=None,
            mask=None):
        """Initialize.

        Arguments:
            stimulus_id: A unique identifier denoting a particular
                stimulus.
                shape=(n_sequence, n_trial)
            class_id: The correct class ID of the stimulus.
                shape = (n_sequence, n_trial)
            is_feedback (optional): A Boolean array indicating if a
                trial provides feedback. By default, all trials are
                assumed to provide feedback.
                shape=(n_sequence, n_trial)
            kind (optional): An array of integers indicating the
                type of trial. This information is useful for grabbing
                subsets of the sequence.
                shape=(n_sequence, n_trial)
            mask (optional): TODO

        """
        self.stimulus_id = self._check_stimulus_id(stimulus_id)
        self.n_sequence = self.stimulus_id.shape[0]
        self.n_trial = self.stimulus_id.shape[1]
        self.class_id = self._check_class_id(class_id)

        if mask is None:
            mask = np.ones([self.n_sequence, self.n_trial], dtype=bool)
        else:
            mask = self._check_mask(mask)
        self.mask = mask

        if is_feedback is None:
            is_feedback = np.ones([self.n_sequence, self.n_trial], dtype=bool)
        else:
            is_feedback = self._check_feedback(is_feedback)
        self.is_feedback = is_feedback

        # TODO move this outside to application code.
        if kind is None:
            kind = np.ones([self.n_sequence, self.n_trial], dtype=int)
            kind[self.is_feedback] = 0
        self.kind = kind

    def _check_stimulus_id(self, stimulus_id):
        """Check `stimulus_id` argument.

        Returns:
            stimulus_id

        Raises:
            ValueError

        """
        # TODO test
        if len(stimulus_id.shape) != 2:
            raise ValueError((
                "The argument `stimulus_id` must be a 2D array of "
                "integers."))

        if not issubclass(stimulus_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be a 2D array of "
                "integers."))

        # TODO check/test non-negative

        return stimulus_id

    def _check_class_id(self, class_id):
        """Check `class_id` argument.

        Returns:
            class_id

        Raises:
            ValueError

        """
        if len(class_id.shape) != 2:
            raise ValueError((
                "The argument `class_id` must be a 2D array of "
                "integers."
            ))

        if not issubclass(class_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `class_id` must be a 2D array of "
                "integers."
            ))

        if not class_id.shape[0] == self.n_sequence:
            raise ValueError((
                "The argument `class_id` must be the same number of "
                "sequences as `stimulus_id`."
            ))

        if not class_id.shape[1] == self.n_trial:
            raise ValueError((
                "The argument `class_id` must be the same number of trials "
                "as `stimulus_id`."
            ))

        # TODO check/test non-negative
        return class_id

    def _check_feedback(self, is_feedback):
        """Check `is_feedback` argument.

        Returns:
            is_feedback

        Raises:
            ValueError

        """
        if not is_feedback.shape[0] == self.n_sequence:
            raise ValueError((
                "The argument `is_feedback` must be the same number of "
                "sequences as `stimulus_id`."
            ))

        if not is_feedback.shape[1] == self.n_trial:
            raise ValueError((
                "The argument `is_feedback` must be the same number of trials "
                "as `stimulus_id`."
            ))

        return is_feedback

    def _check_mask(self, mask):
        """Check `mask` argument.

        Returns:
            mask

        Raises:
            ValueError

        """
        # TODO test
        if len(mask.shape) != 2:
            raise ValueError((
                "The argument `mask` must be a 2D array of "
                "booleans."))

        if not issubclass(mask.dtype.type, np.bool_):
            raise ValueError((
                "The argument `mask` must be a 2D array of "
                "booleans."))

        return mask

    def save(obj):
        """Save method for pickle."""
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        """Load method for pickle."""
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj


class BehaviorSequence(TrialSequence):
    """Class for storing a behavior sequence."""

    def __init__(self, response_time_ms, weight=None, group_id=None):
        """Initialize."""
        self.response_time_ms = self._check_response_time(response_time_ms)
        self.n_sequence = self.response_time_ms.shape[0]
        self.n_trial = self.response_time_ms.shape[1]

        if weight is None:
            weight = np.ones([self.n_sequence, self.n_trial])
        else:
            weight = self._check_weight(weight)
        self.weight = weight

        if group_id is None:
            self.group_id = np.zeros(
                [self.n_sequence, self.n_trial], dtype=int
            )
        else:
            group_id = self._check_group_id(group_id)
        self.group_id = group_id

    def _check_response_time(self, response_time_ms):
        """Check `response_time_ms` argument.

        Returns:
            class_id

        Raises:
            ValueError

        """
        # TODO
        return response_time_ms

    def _check_weight(self, weight):
        """Check `weight` Argument.

        Returns:
            weight

        Raises:
            ValueError

        """
        # TODO
        return weight

    def _check_group_id(self, group_id):
        """Check `group_id` argument.

        Returns:
            group_id

        Raises:
            ValueError

        """
        # TODO
        return group_id


class AFCSequence(BehaviorSequence):
    """Class for storing a sequence of AFC behavior.

    Class makes the necessary assumptions in order to model alternative
    forced choice (AFC) behavior.
    """

    def __init__(self, class_id, response_time_ms, weight=None, group_id=None):
        """Initialize."""
        BehaviorSequence.__init__(
            self, response_time_ms, weight=weight, group_id=group_id
        )
        self.class_id = self._check_class_id(class_id)

    def _check_class_id(self, class_id):
        """Check `class_id` argument.

        Returns:
            class_id

        Raises:
            ValueError

        """
        if len(class_id.shape) != 2:
            raise ValueError((
                "The argument `class_id` must be a 2D array of "
                "integers."))

        if not issubclass(class_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `class_id` must be a 2D array of "
                "integers."))
        return class_id

    def save(obj):
        """Save method for pickle."""
        return (obj.__class__, obj.__dict__)

    def load(cls, attributes):
        """Load method for pickle."""
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj


def stack(seq_list, postpend=True):
    """Combine StimulusSequence objects.

    Since trial sequences may be different lengths, padding is used in
    conjunction with a mask to make all sequences the same length.

    Arguments:
        seq_list: A tuple of StimulusSequence objects to be combined.

    Returns:
        A new StimulusSequence object.

    """
    # Determine the maximum number of trials.
    max_n_trial = 0
    for i_seq in seq_list:
        if i_seq.n_trial > max_n_trial:
            max_n_trial = i_seq.n_trial

    # Grab relevant information from first entry in list.
    is_stim = True
    try:
        class_id = pad_trial(
            seq_list[0].class_id, max_n_trial, postpend=postpend
        )
        mask = pad_trial(
            seq_list[0].mask, max_n_trial, postpend=postpend
        )
        is_feedback = pad_trial(
            seq_list[0].is_feedback, max_n_trial, postpend=postpend
        )
        stimulus_id = pad_trial(
            seq_list[0].stimulus_id, max_n_trial, postpend=postpend
        )
        kind = pad_trial(
            seq_list[0].kind, max_n_trial, postpend=postpend
        )
    except AttributeError:
        is_stim = False
        class_id = pad_trial(seq_list[0].class_id, max_n_trial)
        response_time_ms = pad_trial(seq_list[0].response_time_ms, max_n_trial)

    if is_stim:
        for i_seq in seq_list[1:]:
            curr_stimulus_id = pad_trial(
                i_seq.stimulus_id, max_n_trial, postpend=postpend
            )
            stimulus_id = np.concatenate(
                [stimulus_id, curr_stimulus_id], axis=0
            )

            curr_class_id = pad_trial(
                i_seq.class_id, max_n_trial, postpend=postpend
            )
            class_id = np.concatenate([class_id, curr_class_id], axis=0)

            curr_mask = pad_trial(
                i_seq.mask, max_n_trial, postpend=postpend
            )
            mask = np.concatenate([mask, curr_mask], axis=0)

            curr_is_feedback = pad_trial(
                i_seq.is_feedback, max_n_trial, postpend=postpend
            )
            is_feedback = np.concatenate(
                [is_feedback, curr_is_feedback], axis=0
            )

            curr_kind = pad_trial(
                i_seq.kind, max_n_trial, postpend=postpend
            )
            kind = np.concatenate(
                [kind, curr_kind], axis=0
            )
        seq = StimulusSequence(
            stimulus_id, class_id, is_feedback=is_feedback, kind=kind,
            mask=mask
        )
    else:
        # TODO must handle different types and additional info
        for i_seq in seq_list[1:]:
            curr_class_id = pad_trial(
                i_seq.class_id, max_n_trial, postpend=postpend
            )
            class_id = np.concatenate([class_id, curr_class_id], axis=0)

            curr_response_time_ms = pad_trial(
                i_seq.response_time_ms, max_n_trial, postpend=postpend
            )
            response_time_ms = np.concatenate(
                [response_time_ms, curr_response_time_ms], axis=0
            )
        seq = AFCSequence(class_id, response_time_ms)

    return seq


def pad_trial(x, n_trial_max, postpend=True):
    """Pad data structure."""
    s = list(x.shape)
    n_trial = s[1]
    if n_trial < n_trial_max:
        n_trial_pad = n_trial_max - n_trial
        s[1] = n_trial_pad
        x_pad = np.zeros(s, dtype=x.dtype)

        if postpend:
            x = np.concatenate([x, x_pad], axis=1)
        else:
            x = np.concatenate([x_pad, x], axis=1)
    return x


# def load_sequence(filepath, verbose=0):
#     """Load sequence saved via the save method.

#     The loaded data is instantiated as a psixy.sequence.TrialSequence
#     object.

#     Arguments:
#         filepath: The location of the hdf5 file to load.
#         verbose (optional): Controls the verbosity of printed summary.

#     Returns:
#         Loaded psixy.sequence.TrialSequence object.

#     """
#     seq = pickle.load(filepath)
#     return seq
