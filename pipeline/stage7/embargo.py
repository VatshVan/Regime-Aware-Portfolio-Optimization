import numpy as np

def calculate_embargo_region(unique_dates: np.ndarray, test_end_idx: int, embargo_pct: float) -> tuple:
    """
    Calculates the embargo window size and the exact end date of the embargo period.
    
    Args:
        unique_dates: Sorted array of all unique EventTimes in the dataset.
        test_end_idx: The integer index of the test set's maximum EventTime.
        embargo_pct: The percentage of the total chronological dataset length to embargo.
        
    Returns:
        tuple: (embargo_size_in_dates, embargo_end_date)
    """
    total_dates = len(unique_dates)
    embargo_size = int(total_dates * embargo_pct)
    
    embargo_end_idx = min(test_end_idx + embargo_size, total_dates - 1)
    embargo_end_date = unique_dates[embargo_end_idx]
    
    return embargo_size, embargo_end_date
