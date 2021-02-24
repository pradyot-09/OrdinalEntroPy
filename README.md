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


Development
===========

EntroPy was created and is maintained by `Pradyot Patil <https://pradyot-09.github.io/>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/pradyot-09/OrdinalEntroPy>`_.

Note that this program is provided with **NO WARRANTY OF ANY KIND**. If you can, always double check the results.

Acknowledgement
===============

The package and repository structure is adapted from :

- entropy : ttps://github.com/raphaelvallat/entropy

All the credit goes to the author of this excellently maintained package.
