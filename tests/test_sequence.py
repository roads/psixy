# -*- coding: utf-8 -*-
# Copyright 2019 The PsiXy Authors. All Rights Reserved.
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

"""Module for testing `sequence.py`."""

from pathlib import Path
import pickle
import pytest

import numpy as np
import psixy.sequence


class TestStimulusSequence:
    """Test class StimulusSequence."""

    def test_initialization(self):
        """Test initialization of class."""
        stimulus_id = np.array([0, 1, 2, 3, 4, 5], dtype=int)
        class_id = np.array([0, 0, 1, 0, 1, 1], dtype=int)
        is_feedback = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        seq = psixy.sequence.StimulusSequence(
            stimulus_id, class_id, is_feedback
        )

        np.testing.assert_array_equal(seq.stimulus_id, stimulus_id)
        np.testing.assert_array_equal(seq.class_id, class_id)
        np.testing.assert_array_equal(seq.is_feedback, is_feedback)
        assert seq.n_trial == 6

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        # Create object.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5], dtype=int)
        class_id = np.array([0, 0, 1, 0, 1, 1], dtype=int)
        is_feedback = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        seq_orig = psixy.sequence.StimulusSequence(
            stimulus_id, class_id, is_feedback
        )

        # Save object.
        fn = tmpdir.join('stim_seq_test.hdf5')
        pickle.dump(seq_orig, open(fn, "wb"))

        # Load the saved catalog.
        seq_loaded = pickle.load(open(fn, 'rb'))

        # Check that the loaded object is correct.
        np.testing.assert_array_equal(
            seq_orig.stimulus_id, seq_loaded.stimulus_id
        )
        np.testing.assert_array_equal(
            seq_orig.class_id, seq_loaded.class_id
        )
        np.testing.assert_array_equal(
            seq_orig.is_feedback, seq_loaded.is_feedback
        )
        assert seq_orig.n_trial == seq_loaded.n_trial


class TestAFCSequence:
    """Test class AFCSequence."""

    def test_initialization(self):
        """Test initialization of class."""
        class_id = np.array([1, 1, 1, 0, 1, 1], dtype=int)
        response_time_ms = np.array([4200, 4000, 3800, 3900, 3700, 3700])
        seq = psixy.sequence.AFCSequence(class_id, response_time_ms)

        np.testing.assert_array_equal(seq.class_id, class_id)
        np.testing.assert_array_equal(seq.response_time_ms, response_time_ms)
        assert seq.n_trial == 6

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        # Create object.
        class_id = np.array([1, 1, 1, 0, 1, 1], dtype=int)
        response_time_ms = np.array([4200, 4000, 3800, 3900, 3700, 3700])
        seq_orig = psixy.sequence.AFCSequence(class_id, response_time_ms)

        # Save object.
        fn = tmpdir.join('behav_seq_test.hdf5')
        pickle.dump(seq_orig, open(fn, "wb"))

        # Load the saved catalog.
        seq_loaded = pickle.load(open(fn, 'rb'))

        # Check that the loaded object is correct.
        np.testing.assert_array_equal(
            seq_orig.class_id, seq_loaded.class_id
        )
        np.testing.assert_array_equal(
            seq_orig.response_time_ms, seq_loaded.response_time_ms
        )
        assert seq_orig.n_trial == seq_loaded.n_trial


class TestObservationSequence:
    """Test class ObservationSequence."""

    def test_initialization(self):
        """Test initialization of class."""
        stimulus_id = np.array([0, 1, 2, 3, 4, 5], dtype=int)
        stim_class_id = np.array([0, 0, 1, 0, 1, 1], dtype=int)
        is_feedback = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        stim_seq = psixy.sequence.StimulusSequence(
            stimulus_id, stim_class_id, is_feedback
        )

        behav_class_id = np.array([1, 1, 1, 0, 1, 1], dtype=int)
        response_time_ms = np.array([4200, 4000, 3800, 3900, 3700, 3700])
        behav_seq = psixy.sequence.AFCSequence(
            behav_class_id, response_time_ms
        )

        obs_seq = psixy.sequence.ObservationSequence(
            stim_seq, behav_seq
        )

        np.testing.assert_array_equal(
            obs_seq.stimuli.stimulus_id, stimulus_id
        )
        np.testing.assert_array_equal(
            obs_seq.stimuli.class_id, stim_class_id
        )
        np.testing.assert_array_equal(
            obs_seq.stimuli.is_feedback, is_feedback
        )
        np.testing.assert_array_equal(
            obs_seq.behavior.class_id, behav_class_id
        )
        np.testing.assert_array_equal(
            obs_seq.behavior.response_time_ms, response_time_ms
        )
        assert obs_seq.n_trial == 6

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        # Create object.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5], dtype=int)
        stim_class_id = np.array([0, 0, 1, 0, 1, 1], dtype=int)
        is_feedback = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        stim_seq = psixy.sequence.StimulusSequence(
            stimulus_id, stim_class_id, is_feedback
        )

        behav_class_id = np.array([1, 1, 1, 0, 1, 1], dtype=int)
        response_time_ms = np.array([4200, 4000, 3800, 3900, 3700, 3700])
        behav_seq = psixy.sequence.AFCSequence(
            behav_class_id, response_time_ms
        )

        obs_seq_orig = psixy.sequence.ObservationSequence(
            stim_seq, behav_seq
        )

        # Save object.
        fn = tmpdir.join('obs_seq_test.hdf5')
        pickle.dump(obs_seq_orig, open(fn, "wb"))

        # Load the saved catalog.
        obs_seq_loaded = pickle.load(open(fn, 'rb'))

        # Check that the loaded object is correct.
        np.testing.assert_array_equal(
            obs_seq_orig.stimuli.stimulus_id,
            obs_seq_loaded.stimuli.stimulus_id
        )
        np.testing.assert_array_equal(
            obs_seq_orig.stimuli.class_id,
            obs_seq_loaded.stimuli.class_id
        )
        np.testing.assert_array_equal(
            obs_seq_orig.stimuli.is_feedback,
            obs_seq_loaded.stimuli.is_feedback
        )
        np.testing.assert_array_equal(
            obs_seq_orig.behavior.class_id,
            obs_seq_loaded.behavior.class_id
        )
        np.testing.assert_array_equal(
            obs_seq_orig.behavior.response_time_ms,
            obs_seq_loaded.behavior.response_time_ms
        )
        assert obs_seq_orig.n_trial == obs_seq_loaded.n_trial

    def test_is_correct(self):
        """Test method `is_correct`."""
        stimulus_id = np.array([0, 1, 2, 3, 4, 5], dtype=int)
        stim_class_id = np.array([0, 0, 1, 0, 1, 1], dtype=int)
        is_feedback = np.array([0, 0, 0, 0, 1, 1], dtype=bool)
        stim_seq = psixy.sequence.StimulusSequence(
            stimulus_id, stim_class_id, is_feedback
        )

        behav_class_id = np.array([1, 1, 1, 0, 1, 1], dtype=int)
        response_time_ms = np.array([4200, 4000, 3800, 3900, 3700, 3700])
        behav_seq = psixy.sequence.AFCSequence(
            behav_class_id, response_time_ms
        )

        obs_seq = psixy.sequence.ObservationSequence(
            stim_seq, behav_seq
        )

        is_correct = obs_seq.is_correct()

        np.testing.assert_array_equal(
            is_correct, np.array([0, 0, 1, 1, 1, 1], dtype=bool)
        )


class TestPad:
    """Test function pad_trial."""

    def test_basic(self):
        """Test basic functionality."""
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=int)
        x_pad = psixy.sequence.pad_trial(x, 7)

        x_desired = np.array(
            [[1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 9, 10, 0, 0]], dtype=int
        )

        np.testing.assert_array_equal(
            x_pad, x_desired
        )
