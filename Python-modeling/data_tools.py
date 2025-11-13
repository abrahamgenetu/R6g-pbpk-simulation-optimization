"""
Data Tools for PBPK Model
Generate synthetic data and load experimental CSV files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pbpk_model import PBPKModel, PBPKParameters

def generate_synthetic_dataset(condition: str = 'normoxia',
                               n_replicates: int = 4,
                               noise_level: float = 0.05,
                               save_csv: bool = True):
    """
    Generate synthetic experimental data for R6G perfusion
    
    Args:
        condition: 'normoxia' or 'hyperoxia'
        n_replicates: Number of replicate experiments
        noise_level: Relative noise level (0.05 = 5%)
        save_csv: Whether to save as CSV files
        
    Returns:
        Dictionary with time and concentration data
    """
    print(f"\nGenerating synthetic {condition} dataset...")
    
    # True parameters for each condition
    if condition == 'normoxia':
        true_params = np.array([6.661, 0.0689, 0.0124, -121.415, 53.59])
    elif condition == 'hyperoxia':
        true_params = np.array([7.63, 0.0699, 0.0108, -116.415, 53.59])
    else:
        raise ValueError("Condition must be 'normoxia' or 'hyperoxia'")
    
    # Time points (0 to 22 minutes, 220 points)
    time_points = np.linspace(0, 22, 220)
    
    # Initialize model
    params = PBPKParameters()
    model = PBPKModel(params)
    
    # Generate clean solution
    solution = model.simulate(time_points, true_params)
    clean_concentration = model.get_vascular_concentration(solution)
    
    # Generate replicates with noise
    datasets = {}
    
    for i in range(n_replicates):
        # Add experimental variability
        # Combine Gaussian noise with slight systematic drift
        noise = np.random.normal(0, noise_level, len(time_points))
        drift = (np.random.rand() - 0.5) * 0.02  # ±1% systematic offset
        
        noisy_concentration = clean_concentration * (1 + drift + noise)
        
        # Ensure non-negative
        noisy_concentration = np.maximum(noisy_concentration, 0)
        
        datasets[f'replicate_{i+1}'] = {
            'time': time_points,
            'concentration': noisy_concentration
        }
        
        # Save as CSV
        if save_csv:
            df = pd.DataFrame({
                'time_min': time_points,
                'concentration_uM': noisy_concentration
            })
            
            filename = f'Data/{condition}/{condition}_rat_{i+1}.csv'
            df.to_csv(filename, index=False)
            print(f"  Saved: {filename}")
    
    # Calculate and save mean
    all_conc = np.array([datasets[f'replicate_{i+1}']['concentration'] 
                        for i in range(n_replicates)])
    mean_concentration = np.mean(all_conc, axis=0)
    
    datasets['mean'] = {
        'time': time_points,
        'concentration': mean_concentration
    }
    
    if save_csv:
        df_mean = pd.DataFrame({
            'time_min': time_points,
            'concentration_uM': mean_concentration
        })
        filename = f'Data/{condition}/{condition}_mean.csv'
        df_mean.to_csv(filename, index=False)
        print(f"  Saved: {filename}")
    
    return datasets

def load_replicate_data(condition: str = 'normoxia',
                       replicate_ids: list = [1, 2, 3, 4]) -> dict:
    """
    Load multiple replicate CSV files and compute mean
    
    Args:
        condition: 'normoxia' or 'hyperoxia'
        replicate_ids: List of replicate IDs to load
        
    Returns:
        Dictionary with time, individual replicates, and mean
    """
    print(f"\nLoading {condition} data...")
    
    time_points = None
    all_concentrations = []
    
    for rep_id in replicate_ids:
        filename = f'Data/{condition}/{condition}_rat_{rep_id}.csv'
        try:
            data = pd.read_csv(filename)
            
            if time_points is None:
                time_points = data.iloc[:, 0].values
            
            concentration = data.iloc[:, 1].values
            all_concentrations.append(concentration)
            
            print(f"  Loaded: {filename} ({len(concentration)} points)")
            
        except FileNotFoundError:
            print(f"  Warning: {filename} not found, skipping")
    
    if len(all_concentrations) == 0:
        raise ValueError(f"No data files found for {condition}")
    
    all_concentrations = np.array(all_concentrations)
    mean_concentration = np.mean(all_concentrations, axis=0)
    std_concentration = np.std(all_concentrations, axis=0)
    
    return {
        'time': time_points,
        'replicates': all_concentrations,
        'mean': mean_concentration,
        'std': std_concentration,
        'n_replicates': len(all_concentrations)
    }

def visualize_dataset(data: dict, condition: str = 'normoxia'):
    """Visualize replicate data with mean and error bars"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All replicates
    ax1 = axes[0]
    for i, rep_conc in enumerate(data['replicates']):
        ax1.plot(data['time'], rep_conc, alpha=0.5, linewidth=1, 
                label=f'Replicate {i+1}')
    ax1.plot(data['time'], data['mean'], 'k-', linewidth=2.5, label='Mean')
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Concentration (µM)', fontsize=11)
    ax1.set_title(f'{condition.capitalize()} - All Replicates', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean with error bars
    ax2 = axes[1]
    # Subsample for clarity
    indices = np.arange(0, len(data['time']), 10)
    ax2.errorbar(data['time'][indices], data['mean'][indices], 
                yerr=data['std'][indices], 
                fmt='o-', linewidth=2, markersize=4, capsize=3,
                label='Mean ± SD')
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Concentration (µM)', fontsize=11)
    ax2.set_title(f'{condition.capitalize()} - Mean with Error Bars', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{condition}_dataset_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{condition}_dataset_visualization.png'")
    plt.show()

def compare_conditions():
    """Compare normoxia vs hyperoxia datasets"""
    
    # Load both conditions
    norm_data = load_replicate_data('normoxia')
    hyper_data = load_replicate_data('hyperoxia')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Direct comparison
    ax1 = axes[0]
    ax1.plot(norm_data['time'], norm_data['mean'], 'b-', 
            linewidth=2.5, label='Normoxia')
    ax1.fill_between(norm_data['time'], 
                     norm_data['mean'] - norm_data['std'],
                     norm_data['mean'] + norm_data['std'],
                     alpha=0.3, color='blue')
    
    ax1.plot(hyper_data['time'], hyper_data['mean'], 'r-', 
            linewidth=2.5, label='Hyperoxia')
    ax1.fill_between(hyper_data['time'], 
                     hyper_data['mean'] - hyper_data['std'],
                     hyper_data['mean'] + hyper_data['std'],
                     alpha=0.3, color='red')
    
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Concentration (µM)', fontsize=11)
    ax1.set_title('Normoxia vs Hyperoxia Comparison', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference plot
    ax2 = axes[1]
    difference = hyper_data['mean'] - norm_data['mean']
    ax2.plot(norm_data['time'], difference, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.fill_between(norm_data['time'], 0, difference, 
                     where=(difference > 0), alpha=0.3, color='green',
                     label='Hyperoxia > Normoxia')
    ax2.fill_between(norm_data['time'], 0, difference, 
                     where=(difference <= 0), alpha=0.3, color='red',
                     label='Normoxia > Hyperoxia')
    
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Concentration Difference (µM)', fontsize=11)
    ax2.set_title('Hyperoxia - Normoxia', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('condition_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'condition_comparison.png'")
    plt.show()
    
    # Statistical summary
    print("\n" + "="*70)
    print("CONDITION COMPARISON STATISTICS")
    print("="*70)
    
    for condition, data in [('Normoxia', norm_data), ('Hyperoxia', hyper_data)]:
        print(f"\n{condition}:")
        print(f"  Replicates: {data['n_replicates']}")
        print(f"  Peak concentration: {np.max(data['mean']):.4f} ± {np.std(data['replicates'][:, np.argmax(data['mean'])]):.4f} µM")
        print(f"  Time to peak: {data['time'][np.argmax(data['mean'])]:.2f} min")
        print(f"  Final concentration: {data['mean'][-1]:.4f} ± {data['std'][-1]:.4f} µM")
        print(f"  AUC: {np.trapz(data['mean'], data['time']):.2f} µM·min")

def generate_all_datasets():
    """Generate complete synthetic dataset for both conditions"""
    
    import os
    
    # Create directories
    os.makedirs('Data/normoxia', exist_ok=True)
    os.makedirs('Data/hyperoxia', exist_ok=True)
    
    print("="*70)
    print("GENERATING SYNTHETIC EXPERIMENTAL DATASETS")
    print("="*70)
    
    # Generate normoxia data
    norm_data = generate_synthetic_dataset('normoxia', n_replicates=4)
    visualize_dataset(norm_data, 'normoxia')
    
    # Generate hyperoxia data
    hyper_data = generate_synthetic_dataset('hyperoxia', n_replicates=4)
    visualize_dataset(hyper_data, 'hyperoxia')
    
    # Compare conditions
    compare_conditions()
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  Data/normoxia/normoxia_rat_1.csv to normoxia_rat_4.csv")
    print("  Data/normoxia/normoxia_mean.csv")
    print("  Data/hyperoxia/hyperoxia_rat_1.csv to hyperoxia_rat_4.csv")
    print("  Data/hyperoxia/hyperoxia_mean.csv")

if __name__ == "__main__":
    generate_all_datasets()
    