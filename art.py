# ART algorithm adapted from Neupy version 0.8.2: https://github.com/itdxer/neupy/releases/tag/v0.8.2

from __future__ import division

import numpy as np
import numbers
import inspect
from six import with_metaclass
from abc import ABCMeta
from collections import namedtuple

from scipy.sparse import issparse

Option = namedtuple('Option', 'class_name value')
number_type = (int, float, np.floating, np.integer)

def preformat_value(value):
    if inspect.isfunction(value) or inspect.isclass(value):
        return value.__name__

    elif isinstance(value, (list, tuple, set)):
        return [preformat_value(v) for v in value]

    elif isinstance(value, (np.ndarray, np.matrix)):
        return value.shape

    return value

def format_data(data, is_feature1d=True, copy=False, make_float=True):
    """
    Transform data in a standardized format.

    Notes
    -----
    It should be applied to the input data prior to use in
    learning algorithms.

    Parameters
    ----------
    data : array-like
        Data that should be formatted. That could be, matrix, vector or
        Pandas DataFrame instance.

    is_feature1d : bool
        Should be equal to ``True`` if input data is a vector that
        contains N samples with 1 feature each. Defaults to ``True``.

    copy : bool
        Defaults to ``False``.

    make_float : bool
        If `True` then input will be converted to float.
        Defaults to ``False``.

    Returns
    -------
    ndarray
        The same input data but transformed to a standardized format
        for further use.
    """
    if data is None or issparse(data):
        return data

    if make_float:
        data = asfloat(data)

    if not isinstance(data, np.ndarray) or copy:
        data = np.array(data, copy=copy)

    # Valid number of features for one or two dimensions
    n_features = data.shape[-1]

    if data.ndim == 1:
        data_shape = (n_features, 1) if is_feature1d else (1, n_features)
        data = data.reshape(data_shape)

    return data

def as_tuple(*values):
    """
    Convert sequence of values in one big tuple.

    Parameters
    ----------
    *values
        Values that needs to be combined in one big tuple.

    Returns
    -------
    tuple
        All input values combined in one tuple

    Examples
    --------
    >>> as_tuple(None, (1, 2, 3), None)
    (None, 1, 2, 3, None)
    >>>
    >>> as_tuple((1, 2, 3), (4, 5, 6))
    (1, 2, 3, 4, 5, 6)
    """
    cleaned_values = []
    for value in values:
        if isinstance(value, (tuple, list)):
            cleaned_values.extend(value)
        else:
            cleaned_values.append(value)
    return tuple(cleaned_values)

def asfloat(value):
    """
    Convert variable to 32 bit float number.

    Parameters
    ----------
    value : matrix, ndarray, Tensorfow variable or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray, Tensorfow variable or scalar
        Output would be input value converted to 32 bit float.
    """
    float_type = 'float32'

    if isinstance(value, (np.matrix, np.ndarray)):
        if value.dtype != np.dtype(float_type):
            return value.astype(float_type)

        return value

    # elif isinstance(value, (tf.Tensor, tf.SparseTensor)):
    #     return tf.cast(value, tf.float32)

    elif issparse(value):
        return value

    float_x_type = np.cast[float_type]
    return float_x_type(value)

class BaseProperty:
    """
    Base class for properties.

    Parameters
    ----------
    default : object
        Default property value. Defaults to ``None``.

    required : bool
        If parameter equal to ``True`` and value is not defined
        after initialization then exception will be triggered.
        Defaults to ``False``.

    allow_none : bool
        When value is equal to ``True`` than ``None`` is a valid
        value for the parameter. Defaults to ``False``.

    Attributes
    ----------
    name : str or None
        Name of the property. ``None`` in case if name
        wasn't specified.

    expected_type : tuple or object
        Expected data types of the property.
    """
    expected_type = object

    def __init__(self, default=None, required=False, allow_none=False):
        self.name = None
        self.default = default
        self.required = required
        self.allow_none = allow_none

        if allow_none:
            self.expected_type = as_tuple(self.expected_type, type(None))

    def __set__(self, instance, value):
        if not self.allow_none or value is not None:
            self.validate(value)

        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        if instance is None:
            return

        if self.default is not None and self.name not in instance.__dict__:
            self.__set__(instance, self.default)

        return instance.__dict__.get(self.name, None)

    def validate(self, value):
        """
        Validate properties value

        Parameters
        ----------
        value : object
        """
        if not isinstance(value, self.expected_type):
            availabe_types = as_tuple(self.expected_type)
            availabe_types = ', '.join(t.__name__ for t in availabe_types)
            dtype = value.__class__.__name__

            raise TypeError(
                "Invalid data type `{0}` for `{1}` property. "
                "Expected types: {2}".format(dtype, self.name, availabe_types))

    def __repr__(self):
        classname = self.__class__.__name__

        if self.name is None:
            return '{}()'.format(classname)

        return '{}(name="{}")'.format(classname, self.name)


class WithdrawProperty(object):
    """
    Defines inherited property that needs to be withdrawn.

    Attributes
    ----------
    name : str or None
        Name of the property. ``None`` in case if name
        wasn't specified.
    """
    def __get__(self, instance, owner):
        # Remove itself, to make sure that instance doesn't
        # have reference to this property. Instead user should
        # be able to see default value from the parent classes,
        # but not allowed to assign different value in __init__
        # method.
        #
        # Other part of functionality defined in the
        # ``ConfigMeta`` class.
        del self


class Property(BaseProperty):
    """
    Simple and flexible class that helps identity properties with
    specified type.

    Parameters
    ----------
    expected_type : object
        Valid data types.

    {BaseProperty.Parameters}
    """
    def __init__(self, expected_type=object, *args, **kwargs):
        self.expected_type = expected_type
        super(Property, self).__init__(*args, **kwargs)

class BoundedProperty(BaseProperty):
    """
    Number property that have specified numerical bounds.

    Parameters
    ----------
    minval : float
        Minimum possible value for the property.

    maxval : float
        Maximum possible value for the property.

    {BaseProperty.Parameters}
    """

    def __init__(self, minval=-np.inf, maxval=np.inf, *args, **kwargs):
        self.minval = minval
        self.maxval = maxval
        super(BoundedProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        super(BoundedProperty, self).validate(value)

        if not (self.minval <= value <= self.maxval):
            raise ValueError(
                "Value `{}` should be between {} and {}"
                "".format(self.name, self.minval, self.maxval))


class ProperFractionProperty(BoundedProperty):
    """
    Proper fraction property. Identify all possible numbers
    between zero and one.

    Parameters
    ----------
    {BaseProperty.Parameters}
    """
    expected_type = (float, int)

    def __init__(self, *args, **kwargs):
        super(ProperFractionProperty, self).__init__(
            minval=0, maxval=1, *args, **kwargs)

class NumberProperty(BoundedProperty):
    """
    Float or integer number property.

    Parameters
    ----------
    {BoundedProperty.Parameters}
    """
    expected_type = number_type

class IntProperty(BoundedProperty):
    """
    Integer property.

    Parameters
    ----------
    {BoundedProperty.Parameters}
    """
    expected_type = (numbers.Integral, np.integer)

    def __set__(self, instance, value):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        super(IntProperty, self).__set__(instance, value)

class ExtractParameters(object):
    def get_params(self, deep=False):
        options = {}

        for property_name, option in self.options.items():
            value = getattr(self, property_name)

            # if isinstance(value, tf.Variable):
            #     value = tensorflow_eval(value)

            property_ = option.value
            is_numpy_array = isinstance(value, np.ndarray)

            if hasattr(option.value, 'choices'):
                choices = property_.choices

                if not is_numpy_array and value in choices.values():
                    choices = {v: k for k, v in choices.items()}
                    value = choices[value]

            options[property_name] = value

        return options

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

class BaseConfigurable(ExtractParameters):
    """
    Base configuration class. It help set up and validate
    initialized property values.

    Parameters
    ----------
    **options
        Available properties.
    """
    def __init__(self, **options):
        available_options = set(self.options.keys())
        invalid_options = set(options) - available_options

        if invalid_options:
            clsname = self.__class__.__name__
            raise ValueError(
                "The `{}` object contains invalid properties: {}"
                "".format(clsname, ', '.join(invalid_options)))

        for key, value in options.items():
            setattr(self, key, value)

        for option_name, option in self.options.items():
            if option.value.required and option_name not in options:
                raise ValueError(
                    "Option `{}` is required.".format(option_name))

class SharedDocsMeta(type):
    """
    Meta-class for shared documentation. This class contains
    main functionality that help inherit parameters and methods
    descriptions from parent classes. This class automatically
    format class documentation using basic python format syntax
    for objects.
    """
    def __new__(cls, clsname, bases, attrs):
        new_class = super(SharedDocsMeta, cls).__new__(
            cls, clsname, bases, attrs)

        # if attrs.get('inherit_method_docs', True):
        #     inherit_docs_for_methods(new_class, attrs)

        # if new_class.__doc__ is None:
        #     return new_class

        # class_docs = new_class.__doc__
        # n_indents = find_numpy_doc_indent(class_docs)

        # if n_indents is not None:
        #     new_class.__doc__ = format_docs(new_class, new_class.__mro__)

        return new_class

class ConfigMeta(SharedDocsMeta):
    """
    Meta-class that configure initialized properties. Also it helps
    inheit properties from parent classes and use them.
    """
    def __new__(cls, clsname, bases, attrs):
        new_class = super(ConfigMeta, cls).__new__(cls, clsname, bases, attrs)
        parents = [kls for kls in bases if isinstance(kls, ConfigMeta)]

        if not hasattr(new_class, 'options'):
            new_class.options = {}

        for base_class in parents:
            new_class.options = dict(base_class.options, **new_class.options)

        options = new_class.options

        # Set properties names and save options for different classes
        for key, value in attrs.items():
            if isinstance(value, BaseProperty):
                value.name = key
                options[key] = Option(class_name=clsname, value=value)

            if isinstance(value, WithdrawProperty) and key in options:
                del options[key]

        return new_class

class ConfigABCMeta(ABCMeta, ConfigMeta):
    """
    Meta-class that combines ``ConfigMeta`` and ``abc.ABCMeta``
    meta-classes.
    """

class ConfigurableABC(with_metaclass(ConfigABCMeta, BaseConfigurable)):
    """
    Class that combine ``BaseConfigurable`` class functionality,
    ``ConfigMeta`` and ``abc.ABCMeta`` meta-classes.
    """

class BaseSkeleton(ConfigurableABC):
    """
    Base class for neural network algorithms.

    Methods
    -------
    fit(\*args, \*\*kwargs)
        Alias to the ``train`` method.

    predict(X)
        Predicts output for the specified input.
    """

    def __init__(self, *args, **options):
        super(BaseSkeleton, self).__init__(*args, **options)

        # self.logs.title("Main information")
        # self.logs.message("ALGORITHM", self.__class__.__name__)
        # self.logs.newline()

        # for key, data in sorted(self.options.items()):
        #     formated_value = preformat_value(getattr(self, key))
        #     msg_text = "{} = {}".format(key, formated_value)
        #     self.logs.message("OPTION", msg_text, color='green')

        # self.logs.newline()

    def repr_options(self):
        options = []
        for option_name in self.options:
            option_value = getattr(self, option_name)
            option_value = preformat_value(option_value)

            option_repr = "{}={}".format(option_name, option_value)
            options.append(option_repr)

        return ', '.join(options)

    def __repr__(self):
        class_name = self.__class__.__name__
        available_options = self.repr_options()
        return "{}({})".format(class_name, available_options)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class BaseNetwork(BaseSkeleton):
    """
    Base class for Neural Network algorithms.

    Parameters
    ----------
    step : float
        Learning rate, defaults to ``0.1``.

    show_epoch : int
        This property controls how often the network will display
        information about training. It has to be defined as positive
        integer. For instance, number ``100`` mean that network shows
        summary at 1st, 100th, 200th, 300th ... and last epochs.

        Defaults to ``1``.

    shuffle_data : bool
        If it's ``True`` than training data will be shuffled before
        the training. Defaults to ``True``.

    signals : dict, list or function
        Function that will be triggered after certain events during
        the training.

    {Verbose.Parameters}

    Methods
    -------
    {BaseSkeleton.fit}

    predict(X)
        Propagates input ``X`` through the network and
        returns produced output.

    plot_errors(logx=False, show=True, **figkwargs)
        Using errors collected during the training this method
        generates plot that can give additional insight into the
        performance reached during the training.

    Attributes
    ----------
    errors : list
        Information about errors. It has two main attributes, namely
        ``train`` and ``valid``. These attributes provide access to
        the training and validation errors respectively.

    last_epoch : int
        Value equals to the last trained epoch. After initialization
        it is equal to ``0``.

    n_updates_made : int
        Number of training updates applied to the network.
    """
    step = NumberProperty(default=0.1, minval=0)
    show_epoch = IntProperty(minval=1, default=1)
    shuffle_data = Property(default=False, expected_type=bool)
    #signals = Property(expected_type=object)

    def __init__(self, *args, **options):
        super(BaseNetwork, self).__init__(*args, **options)

        # self.last_epoch = 0
        # self.n_updates_made = 0
        # self.errors = base_signals.ErrorCollector()

        # signals = list(as_tuple(
        #     base_signals.ProgressbarSignal(),
        #     base_signals.PrintLastErrorSignal(),
        #     self.errors,
        #     self.signals,
        # ))

        # for i, signal in enumerate(signals):
        #     if inspect.isfunction(signal):
        #         signals[i] = base_signals.EpochEndSignal(signal)

        #     elif inspect.isclass(signal):
        #         signals[i] = signal()

        # self.events = Events(network=self, signals=signals)

class ART1(BaseNetwork):
    """
    Adaptive Resonance Theory (ART1) Network for binary
    data clustering.

    Notes
    -----
    - Weights are not random, so the result will be
      always reproduceble.

    Parameters
    ----------
    rho : float
        Control reset action in training process. Value must be
        between ``0`` and ``1``, defaults to ``0.5``.

    n_clusters : int
        Number of clusters, defaults to ``2``. Min value is also ``2``.

    {BaseNetwork.Parameters}

    Methods
    -------
    train(X)
        ART trains until all clusters are found.

    predict(X)
        Each prediction trains a new network. It's an alias to
        the ``train`` method.

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> data = np.array([
    ...     [0, 1, 0],
    ...     [1, 0, 0],
    ...     [1, 1, 0],
    ... ])
    >>>>
    >>> artnet = algorithms.ART1(
    ...     step=2,
    ...     rho=0.7,
    ...     n_clusters=2,
    ...     verbose=False
    ... )
    >>> artnet.predict(data)
    array([ 0.,  1.,  1.])
    """
    rho = ProperFractionProperty(default=0.5)
    n_clusters = IntProperty(default=2, minval=2)

    def train(self, X, forms):
        X = format_data(X)

        if X.ndim != 2:
            raise ValueError("Input value must be 2 dimensional, got "
                             "{}".format(X.ndim))

        n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        step = self.step
        rho = self.rho

        if np.any((X != 0) & (X != 1)):
            raise ValueError("ART1 Network works only with binary matrices")

        if not hasattr(self, 'weight_21'):
            self.weight_21 = np.ones((n_features, n_clusters))

        if not hasattr(self, 'weight_12'):
            scaler = step / (step + n_clusters - 1)
            self.weight_12 = scaler * self.weight_21.T

        weight_21 = self.weight_21
        weight_12 = self.weight_12

        if n_features != weight_21.shape[0]:
            raise ValueError("Input data has invalid number of features. "
                             "Got {} instead of {}"
                             "".format(n_features, weight_21.shape[0]))

        classes = np.zeros(n_samples)
        
        # Train network
        for i, p in enumerate(X):
            disabled_neurons = []
            reseted_values = []
            reset = True

            while reset:
                output1 = p
                input2 = np.dot(weight_12, output1.T)
                
                output2 = np.zeros(input2.size)
                input2[disabled_neurons] = -np.inf
                winner_index = input2.argmax()
                output2[winner_index] = 1
                
                expectation = np.dot(weight_21, output2)
                output1 = np.logical_and(p, expectation).astype(int)

                reset_value = np.dot(output1.T, output1) / np.dot(p.T, p)
                reset = reset_value < rho

                if reset:
                    disabled_neurons.append(winner_index)
                    reseted_values.append((reset_value, winner_index))
                
                if len(disabled_neurons) >= n_clusters:
                    # Got this case only if we test all possible clusters
                    reset = False
                    winner_index = None

                if not reset:
                    if winner_index is not None:
                        weight_12[winner_index, :] = (step * output1) / (
                            step + np.dot(output1.T, output1) - 1
                        )
                        weight_21[:, winner_index] = output1
                        
                    else:
                        # Get result with the best `rho`
                        winner_index = max(reseted_values)[1]

                    classes[i] = winner_index

        prototypes = weight_21.T ## TODO: this is not correct, prototypes should contain winning weights_21 for all data points
        return classes, prototypes

    def predict(self, X):
        return self.train(X)
