"""
Dog Head Orientation Labeling Script
=====================================

This script analyzes DeepLabCut nose tracking data to determine where a dog is looking.
Labels: LEFT, RIGHT, STRAIGHT, ELSEWHERE

The algorithm uses:
1. The relative position of nose.tip to the midpoint between nose.left and nose.right
2. Likelihood scores to detect when the dog is not clearly visible

Author: Generated for head orientation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(excel_path: str) -> pd.DataFrame:
    """
    Load and parse the DeepLabCut Excel output file.
    
    Args:
        excel_path: Path to the Excel file containing nose tracking data
        
    Returns:
        DataFrame with properly named columns
    """
    # Skip the header rows and load data
    df = pd.read_excel(excel_path, header=None, skiprows=2)
    
    # Rename columns based on the DeepLabCut structure
    df.columns = [
        'frame', 
        'nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood',
        'nose_right_x', 'nose_right_y', 'nose_right_likelihood',
        'nose_bottom_x', 'nose_bottom_y', 'nose_bottom_likelihood',
        'nose_left_x', 'nose_left_y', 'nose_left_likelihood'
    ]
    
    return df


def calculate_head_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics needed for head orientation classification.
    
    The key metric is the 'tip_offset_ratio' which measures how far the nose tip
    is displaced from the center of the nose (midpoint between left and right nostrils).
    
    - Negative values indicate the nose tip is pointing LEFT
    - Positive values indicate the nose tip is pointing RIGHT  
    - Values near zero indicate looking STRAIGHT
    
    Args:
        df: DataFrame with nose tracking data
        
    Returns:
        DataFrame with added metric columns
    """
    # Calculate the midpoint between nose_left and nose_right (x-axis)
    df['nose_midpoint_x'] = (df['nose_right_x'] + df['nose_left_x']) / 2
    
    # Calculate the width of the nose (distance between left and right points)
    df['nose_width'] = df['nose_right_x'] - df['nose_left_x']
    
    # Calculate how far the nose tip is offset from the midpoint
    # Positive = nose tip is to the RIGHT of center
    # Negative = nose tip is to the LEFT of center
    df['tip_offset'] = df['nose_tip_x'] - df['nose_midpoint_x']
    
    # Normalize by nose width to get a ratio (accounts for distance from camera)
    # Handle division by zero for cases where width is very small
    df['tip_offset_ratio'] = np.where(
        np.abs(df['nose_width']) > 5,  # Only calculate if nose width is reasonable
        df['tip_offset'] / df['nose_width'],
        0
    )
    
    # Calculate average likelihood across all nose points
    df['avg_likelihood'] = (
        df['nose_tip_likelihood'] + 
        df['nose_right_likelihood'] + 
        df['nose_left_likelihood'] + 
        df['nose_bottom_likelihood']
    ) / 4
    
    return df


def classify_orientation(
    row: pd.Series,
    likelihood_threshold: float = 0.3,
    y_margin: float = 2.0
) -> str:
    """
    Classify the head orientation for a single frame based on nose landmark Y positions.
    
    Detection formula:
    - LEFT: nose.right Y < nose.left Y (with margin of 2 pixels)
    - STRAIGHT: nose.right Y ≈ nose.left Y (within margin of 2 pixels)
    - RIGHT: nose.left Y < nose.right Y (with margin of 2 pixels)
    
    Args:
        row: A row from the DataFrame with nose landmark data
        likelihood_threshold: Minimum confidence to consider valid detection
        y_margin: Margin of error in pixels for Y comparison (default: 2.0)
        
    Returns:
        One of: "LEFT", "RIGHT", "STRAIGHT", "ELSEWHERE"
    """
    # If low likelihood, the dog is not clearly visible or looking elsewhere
    if row['avg_likelihood'] < likelihood_threshold:
        return "ELSEWHERE"
    
    # Calculate Y difference between nose.right and nose.left
    y_diff = row['nose_right_y'] - row['nose_left_y']
    
    # Classify based on Y difference
    if abs(y_diff) <= y_margin:
        # nose.right Y ≈ nose.left Y -> STRAIGHT
        return "STRAIGHT"
    elif y_diff < -y_margin:
        # nose.right Y < nose.left Y -> LEFT
        return "LEFT"
    else:
        # nose.left Y < nose.right Y -> RIGHT
        return "RIGHT"


def label_frames(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Apply orientation classification to all frames.
    
    Args:
        df: DataFrame with calculated head metrics
        **kwargs: Parameters for classify_orientation function
        
    Returns:
        DataFrame with 'orientation' column added
    """
    df['orientation'] = df.apply(lambda row: classify_orientation(row, **kwargs), axis=1)
    return df


def aggregate_by_second(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Aggregate frame-level labels to second-level labels.
    Uses majority voting within each second.
    
    Args:
        df: DataFrame with frame-level orientation labels
        fps: Frames per second of the video
        
    Returns:
        DataFrame with one row per second
    """
    # Calculate which second each frame belongs to
    df['second'] = (df['frame'] / fps).astype(int)
    
    # Group by second and get the most common orientation
    results = []
    for second, group in df.groupby('second'):
        # Count orientations
        orientation_counts = group['orientation'].value_counts()
        dominant_orientation = orientation_counts.index[0]
        confidence = orientation_counts.iloc[0] / len(group)
        
        # Get average metrics for this second
        avg_likelihood = group['avg_likelihood'].mean()
        avg_tip_ratio = group['tip_offset_ratio'].mean()
        
        results.append({
            'second': int(second),
            'orientation': dominant_orientation,
            'confidence': round(confidence, 3),
            'avg_likelihood': round(avg_likelihood, 3),
            'avg_tip_offset_ratio': round(avg_tip_ratio, 3),
            'frame_count': len(group)
        })
    
    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, output_path: str):
    """Save results to CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main():
    """Main function to run the head orientation labeling pipeline."""
    
    # Configuration - using project folder structure
    PROJECT_DIR = Path(__file__).parent
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "output"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    INPUT_FILE = DATA_DIR / 'Beauty_T1.xlsx'
    OUTPUT_FILE = OUTPUT_DIR / 'head_orientation_labels.csv'
    VIDEO_DURATION = 9  # seconds
    
    print("=" * 60)
    print("Dog Head Orientation Labeling")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(INPUT_FILE)
    print(f"   Loaded {len(df)} frames")
    
    # Calculate FPS
    fps = len(df) / VIDEO_DURATION
    print(f"   Calculated FPS: {fps:.2f}")
    
    # Calculate metrics
    print("\n2. Calculating head orientation metrics...")
    df = calculate_head_metrics(df)
    
    # Label each frame
    print("\n3. Classifying head orientation for each frame...")
    df = label_frames(
        df,
        likelihood_threshold=0.3,    # Lower threshold to catch more valid frames
        straight_threshold=0.12,     # Based on data analysis
        left_right_threshold=0.12    # Symmetrical threshold
    )
    
    # Show frame-level distribution
    print("\n   Frame-level orientation distribution:")
    print(df['orientation'].value_counts().to_string())
    
    # Aggregate by second
    print("\n4. Aggregating to second-level labels...")
    results = aggregate_by_second(df, fps)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS: Head Orientation by Second")
    print("=" * 60)
    print(results.to_string(index=False))
    
    # Save results
    print("\n5. Saving results...")
    save_results(results, str(OUTPUT_FILE))
    
    # Also save frame-level data for detailed analysis
    frame_output = OUTPUT_DIR / (OUTPUT_FILE.stem + '_frame_level.csv')
    frame_results = df[['frame', 'nose_tip_x', 'nose_tip_y', 'tip_offset_ratio', 
                         'avg_likelihood', 'orientation']].copy()
    frame_results.to_csv(frame_output, index=False)
    print(f"   Frame-level results saved to: {frame_output}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
