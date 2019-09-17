# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
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
    ObservationSequence: TODO

Functions:
    stack: Stack sequences.
    pad_trials: Pad trials.

Todo:
    * Allow for list of lists in response times?

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
        self.version = '0.1.0'


class StimulusSequence(TrialSequence):
    """Class for storing a stimulus sequence."""

    def __init__(
            self, z, class_id, is_real=None, is_feedback=None,
            stimulus_id=None, trial_type=None):
        """Initialize.

        Arguments:
            z: A feature representation.
                shape=(n_sequence, n_trial, n_feature)
            class_id: The correct class ID of the stimulus.
                shape = (n_sequence, n_trial)
            is_real (optional): A boolean array indicating which trials
                are filled with real data.
                shape = (n_sequence, n_trial)
            is_feedback (optional): A Boolean array indicating if a
                trial provides feedback. By default, all trials are
                assumed to provide feedback.
                shape = (n_sequence, n_trial)
            stimulus_id (optional):
                shape = (n_sequence, n_trial)
            trial_type (optional): An array of integers indicating the
                type of trial. This information is useful for grabbing
                subsets of the sequence.
                shape = (n_sequence, n_trial)
        """
        self.z = self._check_z(z)
        self.n_sequence = self.z.shape[0]
        self.n_trial = self.z.shape[1]
        self.n_dim = self.z.shape[2]

        self.class_id = self._check_class_id(class_id)

        if is_real is None:
            is_real = np.ones([self.n_sequence, self.n_trial], dtype=bool)
        else:
            is_real = self._check_is_real(is_real)
        self.is_real = is_real

        if is_feedback is None:
            is_feedback = np.ones([self.n_sequence, self.n_trial], dtype=bool)
        else:
            is_feedback = self._check_feedback(is_feedback)
        self.is_feedback = is_feedback

        # TODO remove stimulus_id attribute? If someone needs it, they can
        # throw it in the feature matrix?
        if stimulus_id is None:
            stimulus_id = -1 * np.ones(
                [self.n_sequence, self.n_trial], dtype=int
            )
        else:
            stimulus_id = self._check_stimulus_id(stimulus_id)
        self.stimulus_id = stimulus_id

        if trial_type is None:
            trial_type = np.ones([self.n_sequence, self.n_trial], dtype=int)
            trial_type[self.is_feedback] = 0
        self.trial_type = trial_type

    def _check_z(self, z):
        """Check `z` argument.

        Returns:
            z

        Raises:
            ValueError

        """
        if len(z.shape) < 3:
            raise ValueError((
                "The argument `z` must be at least a 3D array."
            ))

        return z

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

        return class_id

    def _check_is_real(self, is_real):
        """Check `is_real` argument.

        Returns:
            is_real

        Raises:
            ValueError

        """
        if len(is_real.shape) != 2:
            raise ValueError((
                "The argument `is_real` must be a 2D array of "
                "booleans."))

        if not issubclass(is_real.dtype.type, np.bool_):
            raise ValueError((
                "The argument `is_real` must be a 2D array of "
                "booleans."))

        return is_real

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

    def _check_stimulus_id(self, stimulus_id):
        """Check `stimulus_id` argument.

        Returns:
            stimulus_id

        Raises:
            ValueError

        """
        if len(stimulus_id.shape) != 2:
            raise ValueError((
                "The argument `stimulus_id` must be a 2D array of "
                "integers."))

        if not issubclass(stimulus_id.dtype.type, np.integer):
            raise ValueError((
                "The argument `stimulus_id` must be a 2D array of "
                "integers."))

        return stimulus_id

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

    def __init__(self, response_time_ms):
        """Initialize."""
        self.response_time_ms = self._check_response_time(response_time_ms)
        self.n_sequence = self.response_time_ms.shape[0]
        self.n_trial = self.response_time_ms.shape[1]

    def _check_response_time(self, response_time_ms):
        """Check response time."""
        # TODO
        return response_time_ms


class AFCSequence(BehaviorSequence):
    """Class for storing a sequence of AFC behavior.

    Class makes the necessary assumptions in order to model alternative
    forced choice (AFC) behavior.
    """

    def __init__(self, class_id, response_time_ms):
        """Initialize."""
        BehaviorSequence.__init__(self, response_time_ms)
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


class ObservationSequence(object):
    """Class for containing a sequence of stimulus and behavior."""

    def __init__(self, stimulus_sequence, behavior_sequence):
        """Initialize.

        Arguments:
            stimulus_sequence: A psixy.sequence.StimulusSequence
                object.
            behavior_sequence: A psixy.sequence.BehaviorSequence
                object.
        """
        if not stimulus_sequence.n_trial == behavior_sequence.n_trial:
            raise ValueError((
                "The argument `stimulus_sequence` and `behavior_sequence` "
                "must have the same number of trials."
            ))
        self.n_trial = stimulus_sequence.n_trial
        self.stimuli = stimulus_sequence
        self.behavior = behavior_sequence

    def split(self):
        """Split sequence into stimulus and behavior."""
        return self.stimuli, self.behavior

    def is_correct(self):
        """Determine if each trial is correct."""
        return np.equal(self.stimuli.class_id, self.behavior.class_id)

    def accuracy(self, idx=None):
        """Determine accuracy of sequence."""
        is_correct = self.is_correct()
        if idx is not None:
            is_correct = is_correct[idx]
        acc = np.sum(is_correct) / len(is_correct)
        return acc

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
        z = pad_trial(seq_list[0].z, max_n_trial)
        class_id = pad_trial(seq_list[0].class_id, max_n_trial)
        is_real = pad_trial(seq_list[0].is_real, max_n_trial)
        is_feedback = pad_trial(seq_list[0].is_feedback, max_n_trial)
        stimulus_id = pad_trial(seq_list[0].stimulus_id, max_n_trial)
        trial_type = pad_trial(seq_list[0].trial_type, max_n_trial)
    except AttributeError:
        is_stim = False
        class_id = pad_trial(seq_list[0].class_id, max_n_trial)
        response_time_ms = pad_trial(seq_list[0].response_time_ms, max_n_trial)

    if is_stim:
        for i_seq in seq_list[1:]:
            z = np.concatenate([z, i_seq.z], axis=0)
            class_id = np.concatenate([class_id, i_seq.class_id], axis=0)
            is_real = np.concatenate([is_real, i_seq.is_real], axis=0)
            is_feedback = np.concatenate(
                [is_feedback, i_seq.is_feedback], axis=0
            )
            stimulus_id = np.concatenate(
                [stimulus_id, i_seq.stimulus_id], axis=0
            )
            trial_type = np.concatenate(
                [trial_type, i_seq.trial_type], axis=0
            )
        seq = StimulusSequence(
            z, class_id, is_real=is_real, is_feedback=is_feedback,
            stimulus_id=stimulus_id, trial_type=trial_type
        )
    else:
        for i_seq in seq_list[1:]:
            class_id = np.concatenate([class_id, i_seq.class_id], axis=0)
            response_time_ms = np.concatenate(
                [response_time_ms, i_seq.response_time_ms], axis=0
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
