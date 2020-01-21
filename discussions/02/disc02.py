
import numpy as np
import pandas as pd
import os


def question01(data, labels):
    """
    Returns a dataframe from the
    given data (a dictionary of lists),
    and list of labels.

    >>> data = {'column1': [0,3,5,6], 'column2': [1,3,2,4]}
    >>> labels = 'a b c d'.split()
    >>> out = question01(data, labels)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.index.tolist() == labels
    True
    """

    df = pd.DataFrame(data, index=labels)
    #df['label'] = labels
    #df = df.set_index('label')
    return df


def question02(ser):
    """
    Given a Pandas Series, outputs the
    positions (an index or array) of
    entries of ser that are multiples of 3.

    >>> ser = pd.Series([1, 3, 6, 9])
    >>> out = question02(ser)
    >>> out.tolist() == [1, 2, 3]
    True
    """

    lst = []
    loop = [i for i in ser.values if i%3 == 0]
    for i in loop:
        a = ser[ser == i].index[0]
        lst.append(a)
    return pd.Series(lst)
