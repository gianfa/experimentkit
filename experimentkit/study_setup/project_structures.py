""" Project structures

Project structures
------------------

Here basic directory structures are collected in order to facilitate the study setup. 

"""
from typing import Dict, Union

REPORTS = {
    'reports':{
        'imgs': None
    }
}
"""
Reports directory structure

It contains reports on the experiments run.
"""


EXPERIMENT: Dict[str, Union[dict, None]] = {
    **{
        'data': None,
        'flows': None,
        'imgs': None,
        'models': None,
        'docs': None,
    },
    **REPORTS
}
"""
EXPERIMENT

data: original data or produced ones.
flows: python scripts containing mostly workflows. Frameworks and
    complex funstions are not supposed to stay here.
imgs: all kind of produced images, including plots and tables.
models: models.
docs: documentation on how to make the scripts working.

It includes Reports directory structure.
"""


DATA_SCIENCE: Dict[str, Union[dict, None]] = {
    'project': {
        'data': None,
        'src': None,
        'models': None,
        'docs': None,
        'reports': None,
        'experiments': {
            '_01_desc': {**EXPERIMENT},
        },
    }
}