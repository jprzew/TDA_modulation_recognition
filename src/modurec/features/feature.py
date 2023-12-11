from abc import ABC, abstractmethod
from inspect import signature
from camel_converter import to_snake


class Feature(ABC):
    """Abstract class for features"""

    creator = None

    def __str__(self):
        """Returns string representation of the feature
        the convention is: feature_name__param1_value1__param2_value2"""

        params = signature(self.__init__).parameters

        # Take keys of all parameters that are not default values and are not in *args or **kwargs
        keys = [key for key in params
                if params[key].kind not in {params[key].VAR_KEYWORD,
                                            params[key].VAR_POSITIONAL} and
                params[key].default != getattr(self, key)]

        values = [self.__dict__[key] for key in keys]
        string = to_snake(self.__class__.__name__)

        if keys:
            string += '__' + '_{}__'.join(keys) + '_{}'
            string = string.format(*values)

        return string

    def values(self):
        return self.df[str(self)]

    @abstractmethod
    def compute(self):
        pass
