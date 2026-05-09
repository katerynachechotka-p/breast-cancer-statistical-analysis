"""
utils.py
Custom statistical functions implemented from first principles.
Used across regression_analysis.ipynb and statistical_inference.ipynb.
"""

import numpy as np


def my_mean(data):
    """Calculate arithmetic mean."""
    sumtotal = 0
    n = len(data)
    for element in data:
        sumtotal += element
    return sumtotal / n


def my_median(data):
    """Calculate median."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        middle_1, middle_2 = n // 2 - 1, n // 2
        return (sorted_data[middle_1] + sorted_data[middle_2]) / 2


def my_count(data, item):
    """Count occurrences of item in data."""
    occurences = 0
    for element in data:
        if element == item:
            occurences += 1
    return occurences


def my_mode(data):
    """Calculate mode (returns list if multiple modes)."""
    max_count = 0
    max_items = []
    for element in data:
        current_count = my_count(data, element)
        if current_count > max_count:
            max_count = current_count
            max_items = [element]
        elif current_count == max_count and element not in max_items:
            max_items.append(element)
    if max_count == 1:
        return None
    return max_items if len(max_items) > 1 else max_items[0]


def my_sample_SD(data):
    """Calculate sample standard deviation (Bessel's correction: n-1)."""
    sum_of_squares = 0
    n = len(data)
    mean = my_mean(data)
    for element in data:
        sum_of_squares += (element - mean) ** 2
    return (sum_of_squares / (n - 1)) ** 0.5


def my_range(data):
    """Calculate range."""
    sorted_data = sorted(data)
    return sorted_data[-1] - sorted_data[0]


def my_skewness(data):
    """Calculate skewness using algebraic formula."""
    n = len(data)
    if n < 3:
        raise ValueError("Skewness requires at least 3 data points.")
    mean = my_mean(data)
    std = my_sample_SD(data)
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skewness


def my_kurtosis(data):
    """Calculate excess kurtosis (normal distribution = 0)."""
    n = len(data)
    mean = my_mean(data)
    std = my_sample_SD(data)
    fourth_moment = np.sum((data - mean) ** 4) / n
    kurtosis = fourth_moment / (std ** 4)
    excess_kurtosis = kurtosis - 3
    return excess_kurtosis
