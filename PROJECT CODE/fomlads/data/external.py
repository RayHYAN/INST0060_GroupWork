import numpy as np
import pandas as pd
import csv

def import_for_classification(
        ifname, input_cols=None, target_col=None, classes=None):
    """
    Imports the iris data-set and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    returns
    -------
    inputs -- the data as a numpy.array object  
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    dataframe = pd.read_csv(ifname)
    print("dataframe.columns = %r" % (dataframe.columns,) )
    N = dataframe.shape[0]
    # if no target name is supplied we assume it is the last colunmn in the 
    # data file
    if target_col is None:
        target_col = dataframe.columns[-1]
        potential_inputs = dataframe.columns[:-1]
    else:
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    # if no classes are specified use all in the dataset
    if classes is None:
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
        classes = class_values.unique()
    else:
        # construct a 1d array of the rows to keep
        to_keep = np.zeros(N,dtype=bool)
        for class_name in classes:
            to_keep |= (dataframe[target_col] == class_name)
        # now keep only these rows
        dataframe = dataframe[to_keep]
        # there are a different number of dat items now
        N = dataframe.shape[0]
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
    print("classes = %r" % (classes,))
    # We now want to translate classes to targets, but this depends on our 
    # encoding. For now we will perform a simple encoding from class to integer.
    targets = np.empty(N)
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets[is_class] = class_id
    #print("targets = %r" % (targets,))

    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object
    inputs = dataframe[input_cols].values
    return inputs, targets, input_cols, classes


def import_for_regression(
        ifname, input_cols=None, target_col=None, delimiter=';', header=None):
    """
    Imports data from csv file assuming that all data is real valued.

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    delimiter -- delimiter/separator for data entries in a line

    returns
    -------
    inputs -- the data as a numpy.array object  
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    target_col -- name of target column
    """
    # if no file name is provided then use synthetic data
    dataframe = pd.read_csv(ifname, sep=delimiter, header=header)
    N = dataframe.shape[0]
    # if no target name is supplied we assume it is the last colunmn in the 
    # data file
    if target_col is None:
        target_col = dataframe.columns[-1]
        potential_inputs = dataframe.columns[:-1]
    elif type(target_col) is type(""):
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    else:
        raise ValueError("Integer columns not yet supported")
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object (don't be fooled by the name of the method as_matrix it does
    # not produce a numpy matrix object
    inputs = dataframe[input_cols].as_matrix()
    targets = dataframe[target_col].as_matrix()
    return inputs, targets, input_cols, target_col

def import_data_simple(ifname, delimiter=None, header=False, columns=None):
    """
    Imports a tab/comma/semi-colon/... separated data file as an array of 
    floating point numbers. If the import file has a header then this should
    be specified, and the field names will be returned as the second argument.

    parameters
    ----------
    ifname -- filename/path of data file.
    delimiter -- delimiter of data values
    header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)

    returns
    -------
    data_as_array -- the data as a numpy.array object  
    field_names -- if file has header, then this is a list of strings of the
      the field names imported. Otherwise, it is a None object.
    """
    if delimiter is None:
        delimiter = '\t'
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=delimiter)
        # if the data has a header line we want to avoid trying to import it.
        # instead we'll print it to screen
        if header:
            field_names = next(datareader)
            print("Importing data with field_names:\n\t" + ",".join(field_names))
        else:
            # if there is no header then the field names is a dummy variable
            field_names = None
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            # for each row of data only take the columns we are interested in
            if not columns is None:
                row = [row[c] for c in columns]
            # now store in our data list
            data.append(row)
        print("There are %d entries" % len(data))
        print("Each row has %d elements" % len(data[0]))
    # convert the data (list object) into a numpy array.
    data_as_array = np.array(data).astype(float)
    if not columns is None and not field_names is None:
        # thin the associated field names if needed
        field_names = [field_names[c] for c in columns]
    # return this array to caller (and field_names if provided)
    return data_as_array, field_names


def import_1d_regression_data(
        ifname, input_col=0, target_col=1, **kwargs):
    """
    Imports 1d regression data (univariate input and target) from a
     tab/comma/semi-colon/... separated data file.

    parameters
    ----------
    ifname -- filename/path of data file.
    input_col -- the index of column used for inputs
    target_col -- the index of column used for targets
    <other keyword arguments supported by pandas read_csv function>
        See the use of **kwargs

    returns
    -------
    inputs -- input values (1d array)  
    targets -- target values (1d array)
    """
    # a dataframe object with all data from file
    df = pd.read_csv(ifname, **kwargs)
    # extract column index input_col as inputs
    inputs = df.iloc[:,input_col].to_numpy()
    # extract column index target_col as targets
    targets = df.iloc[:,target_col].to_numpy()
    return inputs, targets


