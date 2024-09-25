# Objectives

The learning objectives of this assignment are to:
1. implement feed-forward prediction for a simple neural network
2. implement training via back-propagation for a simple neural network

Helpful references:

* [Python (version 3.8 or higher)](https://www.python.org/downloads/)
* [numpy](http://www.numpy.org/)
* [pytest](https://docs.pytest.org/)


# Write your code

You will be editing the file `nn.py`by copying the content into a new notebook and naming it as "nn". 
Once all codes are written download this notebook as "nn.ipynb" and "nn.py" files. These are your solution files.
**Do not add or edit any other files.**

You will implement a simple neural network with sigmoid activations everywhere.
This will include making predictions with a network via forward-propagation, and
training the network via gradient descent, with gradients calculated using
back-propagation.

You should read the documentation strings (docstrings) in each of methods in
`nn.py`, and implement the methods as described.
Write your code below the docstring of each method by replacing ##YOUR CODE HERE##;
**do not delete the docstrings**.

The following objects and functions may come in handy:
* [numpy.ndarray.dot](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html)
* [numpy.ndarray.T](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html)
* [numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
* [scipy.special.expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)

# Test your code for correctness

The tests in `test_nn.py` check that each method behaves as expected.
To run all the provided tests, run ``pytest`` from the directory containing
``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py FFFFF                                                         [100%]

=================================== FAILURES ===================================
...
============================== 5 failed in 0.65s ===============================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 5 items

test_nn.py .....                                                         [100%]

============================== 5 passed in 0.47s ===============================
```

