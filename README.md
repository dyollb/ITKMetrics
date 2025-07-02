# ITKMetrics

[![Build Status](https://github.com/dyollb/ITKMetrics/workflows/Build,%20test,%20package/badge.svg)](https://github.com/dyollb/ITKMetrics/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dyollb/ITKMetrics/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/itk-metrics.svg)](https://badge.fury.io/py/itk-metrics)
<img src="https://img.shields.io/pypi/dm/itk-metrics.svg?label=pypi%20downloads&logo=python&logoColor=green"/>
<img src="https://img.shields.io/badge/python-%203.7%20|%203.8%20|%203.9%20|%203.10%20|%203.11%20-3776ab.svg"/>

## Overview

This is a module for the Insight Toolkit ([ITK](https://github.com/InsightSoftwareConsortium/ITK)). The module includes metrics not included in the ITK proper.

```python
    import itk

    labels = itk.imread('path/to/labelmap.mha').astype(itk.US)
    D = labels.ndim

    ImageType = type(labels)
```

## Installation

To install the binary Python packages:

```shell
  python -m pip install itk-metrics
```
