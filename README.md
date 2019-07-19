# PsiXy: A Psychological Category Learning Package

## Purpose
PsiXy provides the computational tools to model human category learning.

## Installation
There are two ways to install PsiXy:

1. Install from PyPI using pip: `pip install psixy` (coming soon)
2. Clone from Git Hub: https://github.com/roads/psixy.git (coming soon)

PsiXy also requires TensorFlow, which is not installed automatically since users will want to specify whether they use a GPU enabled version of TensorFlow.

## Quick Start
There is one predefined models to choose from:

1. ALCOVE

Once you have selected a model, you must provide TODO pieces of information in order to fit the model.

1. Behavioral observations.
2. ?

```python
from psixy import datasets, models

# Load some observations.
(obs, catalog) = datasets.load_dataset()
# Initialize a model.
model = ALCOVE()
# Fit the model using observations.
mdoel.fit(obs)
# Save the fitted model.
mdoel.save('my_model.h5')
```

## Trials and Observations
Inference is performed by fitting a model to a set of observations. In this package, a single observation is comprised of TODO...

In the simplest case, an observation is obtained from a trial consisting of TODO...

## Modules
* `models` - A set of pre-defined pscyhological embedding models.
* `trials` - Data structure used for trials and observations.
* `utils` - Utility functions.

## Support

## Authors
* Brett D. Roads
* Michael C. Mozer
* Bradley C. Love
* See also the list of contributors who participated in this project.

## What's in a name?
The name PsiXy (pronounced *sigh x y*) is meant to serve as shorthard for *psychological category learning*. The greek letter Psi is often used to represent the field of psychology. The matrix variable **X** and vector variable **y** are often used in machine learning to denote a set of input features and labels respectively.

## Licence
This project is licensed under the Apache Licence 2.0 - see the LICENSE.txt file for details.
