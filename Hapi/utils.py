import datetime
import os
import pickle
import sys


def save_obj(obj, saved_name):
    """save_obj.

    this function is used to save any python object to your hard desk

    Parameters
    ----------
    obj: [Any]
        any python object
    saved_name: [str]
        name of the object

    Returns
    -------
    the object will be saved to the given path/current working directory
    with the given name

    Examples
    --------
    >>> path = "path/to/your/disk"
    >>> data={"key1":[1,2,3,5],"key2":[6,2,9,7]}
    >>> save_obj(data, f'{path}/flow_acc_table')
    """
    with open(saved_name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(saved_name):
    """load_obj.

        this function is used to save any python object to your hard desk

    Parameters
    ----------
    1-saved_name:
        ['String'] name of the object

    Returns
    -------
    the object will be loaded

    Examples
    --------
    >>> path = r"c:\my_computer\files"
    >>> load_obj(f'{path}/flow_acc_table')
    """
    with open(saved_name + ".pkl", "rb") as f:
        return pickle.load(f)


def dateformated(x):
    """dateformated.

    this function converts the the date read from a list to a datetime format

    Parameters
    ----------
    x: [list]
        is a list of tuples of string date read from database

    Returns
    -------
        list od dates as a datetime format  YYYY-MM-DD HH:MM:SS
    """
    x = [i[0] for i in x]
    #
    x1 = []
    for i in x:
        if len(i) == 19:
            x1.append(
                datetime.datetime(
                    int(i[:4]),
                    int(i[5:7]),
                    int(i[8:10]),
                    int(i[11:13]),
                    int(i[14:16]),
                    int(i[17:18]),
                )
            )
    #        elif len(i)==13:
    #            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(0),int(0) ))
    #        else:
    #            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(0),int(0),int(0) ))
    #    del i,x
    return x1


def printWaitBar(i, total, prefix="", suffix="", decimals=1, length=100, fill=" "):
    """This function will print a waitbar in the console.

    Parameters:
    i:
        Iteration number
    total:
        Total iterations
    fronttext:
        Name in front of bar
    prefix:
        Name after bar
    suffix:
        Decimals of percentage
    length:
        width of the waitbar
    fill:
        bar fill
    """
    # Adjust when it is a linux computer
    if os.name == "posix" and total == 0:
        total = 0.0001

    percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(total)))
    filled = int(length * i // total)
    bar = fill * filled + "-" * (length - filled)

    sys.stdout.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    if i == total:
        print()


def class_method_parse(initial_args):
    """check and assign values of parameters to the self object.

        check values of a method and assign the valuse of the parameters to the self object (the method has self/cls
        as first parameter)

    Parameters
    ----------
    initial_args: [Dict]
        dictionary contains all the parameters of the function, positional and key word parameters, each parameter is a
        key(i.e 'name' in the below example), and the value is a dict that has at least a key called "type",
        and a value that is an available data type in python, (i.e 'name' : {"type": str}),
        - If the parameter has a default value, the dict has to have another key: value i.e "default": <any value>
        - if there is no "default" key in the parameter dict, the default value will be taken None
        >>> initial_args = {
        >>>     'name' : {"type": str},
        >>>     'version' : {"default": 3, "type": int}
        >>> }

    Returns
    -------
    assign the valuse of the parameters to the self object
    """

    def apply_func(func):
        """apply the function that the decorator preceded.

        Parameters
        ----------
        func: [function]
            the function that the decorator precedes

        Returns
        -------
        returns the same outputs of the input function
        """

        def wrapper(*args, **kwargs):

            self = args[0]
            # get wrong kwargs
            wrong_kwargs = set(kwargs) - set(initial_args)
            if len(wrong_kwargs) > 0:
                print(initial_args)
                raise KeyError(f"Invalid parameter {wrong_kwargs}")

            for key, val in initial_args.items():
                # if the parameter is given by user
                if key in kwargs.keys():
                    default = initial_args.get(key)
                    # check the type
                    key_type = default.get("type")
                    # make the type as a list
                    if not isinstance(key_type, list):
                        key_type = [key_type]
                    # get the given value
                    val = kwargs.get(key)
                    if type(val) in key_type:
                        # set the given value
                        setattr(self, key, val)
                    else:
                        raise TypeError(
                            f"The parameter {key} should be of type {key_type}"
                        )
                else:
                    # positional args
                    if "default" in val.keys():
                        setattr(self, key, val.get("default"))

            res = func(*args, **kwargs)
            return res

        return wrapper

    return apply_func


def class_attr_initialize(attributes):
    """check and assign values of parameters to the self object.

        check values of a method and assign the valuse of the parameters to the self object
        (the method has self/cls
        as first parameter)

    Parameters
    ----------
    initial_args: [Dict]
        dictionary contains all the parameters of the function, positional and key word
        parameters, each parameter is a key(i.e 'name' in the below example), and the
        value is a dict that has at least a key called "type", and a value that is an
        available data type in python, (i.e 'name' : {"type": str}),
        - If the parameter has a default value, the dict has to have another
        key: value i.e "default": <any value>
        >>> initial_args = {
        >>>     'name' : {"type": str},
        >>>     'version' : {"default": 3, "type": int}
        >>> }

    Returns
    -------
    assign the valuse of the parameters to the self object
    """

    def apply_func(func):
        """apply the function that the decorator preceded.

        Parameters
        ----------
        func: [function]
            the function that the decorator precedes

        Returns
        -------
        returns the same outputs of the input function
        """

        def wrapper(*args, **kwargs):
            self = args[0]
            # initialize attributes
            for key, val in attributes.items():
                setattr(self, key, val)

            func(*args, **kwargs)

        return wrapper

    return apply_func
