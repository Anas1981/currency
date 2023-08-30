import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two arrays.

    Parameters:
        - y_true: Array of true values.
        - y_pred: Array of predicted values.

    Returns:
        - MAPE: Mean Absolute Percentage Error.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(absolute_percentage_errors) * 100  # Multiply by 100 to get the percentage
    return mape


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between two arrays.

    Parameters:
        - y_true: Array of true values.
        - y_pred: Array of predicted values.

    Returns:
        - MAE: Mean Absolute Error.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    absolute_errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
    mae = sum(absolute_errors) / len(absolute_errors)
    return mae

