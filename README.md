[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/pradyot-09/OrdinalEntroPy)](https://github.com/pradyot-09/OrdinalEntroPy/blob/main/LICENSE)
[![Build](https://travis-ci.com/pradyot-09/OrdinalEntroPy.svg?branch=main&status=created)](https://travis-ci.com/github/pradyot-09/OrdinalEntroPy)
![stars](https://img.shields.io/github/stars/pradyot-09/OrdinalEntroPy)
# OrdinalEntroPy
OrdinalEntroPy is a Python 3 package providing several time efficient, ordinal pattern based entropy algorithms for computing the complexity of one-dimensional time-series. 
The package consists of following entropy methods:

- [Permutation Entropy (PE)](https://www.semanticscholar.org/paper/Permutation-entropy:-a-natural-complexity-measure-Bandt-Pompe/04de9ce062c6ac999fa009b9c264da20a8d8a282) 
- [Weighted Permutation Entropy (WPE)](https://pubmed.ncbi.nlm.nih.gov/23496595/)
- [Reverse Permutation Entropy (RPE)](https://epub.ub.uni-greifswald.de/frontdoor/deliver/index/docId/2794/file/entropy-19-00197.pdf)
- [Dispersion Entropy (DE)](https://www.semanticscholar.org/paper/Dispersion-Entropy:-A-Measure-for-Time-Series-Rostaghi-Azami/43a842555910bfb1c301bc7ff139d2ffabad19f7)
- [Reverse Dispersion Entropy (RDE)](https://pubmed.ncbi.nlm.nih.gov/31783659/)
- [Reverse Weighted Dispersion Entropy (RWDE)]

Installation
============

important:

  Currently OrdinalEntroPy is not part of pip repository, therefore you cannot install it using pip or conda.

```shell

  git clone https://github.com/pradyot-09/OrdinalEntroPy.git /
  cd OrdinalEntroPy/
  pip install -r requirements.txt
  python setup.py develop
```  
  **Dependencies**

- `numpy <https://numpy.org/>`
- `scipy <https://www.scipy.org/>`


Functions
============

code : 
```python

from OrdinalEntroPy import *
import numpy as np
np.random.seed(1234567)
x = np.random.rand(3000)
print(PE(x, order=3, normalize=True))                        # Permutation entropy
print(WPE(x, order=3, normalize=True))                       # Weighted Permutation Entropy
print(RPE(x, order=3, delay=1, normalize=True))              # Reverse Permutation Entropy
print(DE(x, order=3,classes=3, normalize=True))              # Dispersion Entropy
print(RDE(x, order=3,classes=3,delay=1,normalize=True))      # Reverse Dispersion Entropy
print(RWDE(x, order=3,classes=3,delay=1,normalize=True))     # Reverse Weighted Dispersion Entropy

```
output entropy value :
```
0.9995858289645746
0.9996533403383996
0.0002963060541583906
0.9830685145488814
0.00418284021851621
0.026268994085565402
```


Development
===========

OrdinalEntroPy was created and is maintained by `Pradyot Patil <https://pradyot-09.github.io/>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/pradyot-09/OrdinalEntroPy>`.

Note that this program is provided with **NO WARRANTY OF ANY KIND**. If you can, always double check the results.

Acknowledgement
===============

The package and repository structure is adapted from :

- entropy : <https://github.com/raphaelvallat/entropy>

All the credit goes to the author of this excellently maintained package.
