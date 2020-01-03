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

"""Module of utility functions.

Classes:
    ProgressBar: A simple command line progress bar.

Functions:
    TODO: ...

"""

import datetime
import time

import numpy as np


class ProgressBar(object):
    """Display a progress bar in terminal."""

    def __init__(
            self, total, prefix='', suffix='', decimals=1, length=100,
            fill='█'):
        """Initialize.

        Arguments:
            iteration: Integer indicating current iteration.
            total: Integer indicating total iterations.
            prefix (optional): String that is used as prefix.
            suffix (optional): String that is used as suffix.
            decimals (optional): Integer indicating a positive number
                of decimals in percent complete.
            length (optional): Integer indicating the character length
                of the progress bar.
            fill (optional): String indicating the bar fill character.
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.n_call = 0
        self.start_s = 0
        self.total_s = 0

    def _start(self):
        """Start timer."""
        self.start_s = time.time()

    def _stop(self):
        """Stop time."""
        self.total_s = time.time() - self.start_s

    def update(self, iteration):
        """Update progress bar to display current iteration."""
        # Start time if this is the first call.
        if self.n_call == 0:
            self._start()
        self.n_call = self.n_call + 1

        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (iteration / float(self.total))
        )

        elapsed_time = time.time() - self.start_s
        if iteration == 0:
            time_per_iter = 0.0
        else:
            time_per_iter = elapsed_time / iteration
        eta_s = np.round((self.total - iteration) * time_per_iter)
        eta_str = str(datetime.timedelta(seconds=eta_s))

        filledLength = int(self.length * iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print(
            '\r    {0} |{1}| {2}%% {3} | ETA: {4}'.format(
                self.prefix, bar, percent, self.suffix, eta_str
            ), end='\r'
        )
        # Print a new line on completion.
        if iteration == self.total:
            self._stop()
            print()
            total_str = str(
                datetime.timedelta(seconds=np.round(self.total_s))
            )
            print('    Elapsed time: {0}'.format(total_str))


def softmax(x, axis=0):
    """Return numerically stable softmax.

    Takes advantage of the fact that softmax(x) = softmax(x + c).

    Arguments:
        x: An numpy ndarray of values.
        axis (optional): The axis to normalize along.

    Returns:
        c: Softmax values.

    """
    val_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - val_max)
    denom = np.sum(ex, axis=axis, keepdims=True)
    c = ex / denom
    return c
