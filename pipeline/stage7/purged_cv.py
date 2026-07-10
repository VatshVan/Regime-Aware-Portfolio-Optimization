import numpy as np
import pandas as pd
from typing import List, Tuple
from . import embargo

class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, labels_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates purged and embargoed train/test indices.
        
        Args:
            labels_df: DataFrame with 'EventTime' and 'ExitTime'
            
        Returns:
            List of (train_indices, test_indices) arrays
        """
        
        # 1. Identify all unique dates sorted chronologically
        unique_dates = np.sort(labels_df['EventTime'].unique())
        
        # 2. Split unique dates into n_splits contiguous chunks
        date_indices = np.arange(len(unique_dates))
        test_date_chunks = np.array_split(date_indices, self.n_splits)
        
        # Map original DataFrame index to numerical positions (0 to N-1)
        row_positions = np.arange(len(labels_df))
        event_times = labels_df['EventTime'].values
        exit_times = labels_df['ExitTime'].values
        
        # Calculate embargo size (in number of unique dates)
        embargo_size = int(len(unique_dates) * self.embargo_pct)
        
        folds = []
        
        for chunk in test_date_chunks:
            if len(chunk) == 0:
                continue
                
            test_start_date = unique_dates[chunk[0]]
            test_end_date = unique_dates[chunk[-1]]
            
            # Test indices: labels where EventTime falls within the test date chunk
            test_mask = (event_times >= test_start_date) & (event_times <= test_end_date)
            test_idx = row_positions[test_mask]
            
            # --- PURGING ---
            # Remove any training labels whose holding period overlaps the test window
            # A label overlaps if its EventTime <= test_end_date AND its ExitTime >= test_start_date
            overlap_mask = (event_times <= test_end_date) & (exit_times >= test_start_date)
            
            # --- EMBARGO ---
            # Training labels immediately following the test set might suffer from serial correlation.
            # We embargo (remove) training labels that start within `embargo_size` dates after the test set.
            embargo_size, embargo_end_date = embargo.calculate_embargo_region(
                unique_dates, chunk[-1], self.embargo_pct
            )
            
            # A label is embargoed if its EventTime is strictly after the test set, but before the embargo end
            embargo_mask = (event_times > test_end_date) & (event_times <= embargo_end_date)
            
            # Final Train Mask: Not in test, not overlapping, not embargoed
            train_mask = ~(test_mask | overlap_mask | embargo_mask)
            train_idx = row_positions[train_mask]
            
            folds.append((train_idx, test_idx))
            
        return folds
