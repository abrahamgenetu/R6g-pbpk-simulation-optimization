"""
PBPK Model: R6G Lung Perfusion Kinetics
Physiologically-Based Pharmacokinetic Modeling with Parameter Optimization

Author: Abraham
Date: November 2025
Purpose: Demonstrate pharmacokinetic modeling and optimization expertise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares, differential_evolution
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json

@dataclass
class PBPKParameters:
    """Parameters for R6G PBPK model"""
    # Optimizable parameters
    k2_bar: float = 6.661      # Binding rate constant (min^-1)
    kminus2: float = 0.0689    # Unbinding rate constant (min^-1)
    kd3: float = 0.0124        # Mitochondrial dissociation constant (µM)
    deltam: float = -121.415   # Mitochondrial membrane potential (mV)
    ps1: float = 53.59         # Permeability-surface area (ml/min)
    
    # Fixed physiological parameters
    Ve: float = 0.85           # Vascular volume (ml)
    Vc: float = 1.0            # Cytoplasmic volume (ml)
    Vm: float = 0.02           # Mitochondrial volume fraction
    Vtub: float = 4.0          # Tubing volume (ml)
    
    # Fixed biophysical parameters
    alpha: float = 0.0374158   # ZF/RT (mV^-1)
    Be: float = 0.5            # BSA concentration (µM)
    kd1: float = 0.32          # Vascular dissociation constant (µM)
    ps2: float = 1.123         # Mitochondrial PS (ml/min)
    
    # Experimental conditions
    F: float = 10.0            # Flow rate (ml/min)
    vmaxkm: float = 0.0        # Efflux rate (ml/min)
    deltap: float = -43.0      # Plasma membrane potential (mV)
    deltam_un: float = -0.1    # Uncoupled mitochondrial potential (mV)

class PBPKModel:
    """R6G lung perfusion PBPK model"""
    
    def __init__(self, params: PBPKParameters):
        self.params = params
        
    def ode_system(self, y: np.ndarray, t: float, params_opt: np.ndarray) -> np.ndarray:
        """
        ODE system for R6G PBPK model
        
        States:
        y[0]: Ctub  - Tubing concentration (µM)
        y[1]: Ce_bar - Vascular total concentration (µM)
        y[2]: Cc - Cytoplasmic free concentration (µM)
        y[3]: Cm - Mitochondrial free concentration (µM)
        y[4]: CcBc - Cytoplasmic bound concentration (µM)
        """
        p = self.params
        k2_bar, kminus2, kd3, deltam, ps1 = params_opt
        
        # Calculate apparent mitochondrial volume
        V3 = p.Vm * p.Vc * (1 + (1 / kd3))
        
        # Determine experimental phase and input concentration
        if t >= 0 and t <= 10:  # Loading phase
            Cin = 0.25
            current_deltam = deltam
        elif t > 10 and t < 15.5:  # Wash phase
            Cin = 0.0
            current_deltam = deltam
        else:  # Uncoupler phase (t >= 15.5)
            Cin = 0.0
            current_deltam = p.deltam_un
        
        # Free vascular concentration
        Ce = y[1] / (1 + p.Be / p.kd1)
        
        # Membrane fluxes
        # J1: Flux across plasma membrane
        exp_deltap = np.exp(-p.alpha * p.deltap)
        J1 = -((p.alpha * ps1 * p.deltap) / (exp_deltap - 1)) * \
             (exp_deltap * Ce - y[2])
        
        # J2: Flux across mitochondrial membrane
        exp_deltam = np.exp(-p.alpha * current_deltam)
        J2 = -((p.alpha * p.ps2 * current_deltam) / (exp_deltam - 1)) * \
             (exp_deltam * y[2] - y[3])
        
        # ODEs
        dCtub = (1.0 / p.Vtub) * p.F * (Cin - y[0])
        dCe_bar = (1 / p.Ve) * (-J1 + p.vmaxkm * y[2] + p.F * (y[0] - y[1]))
        dCc = kminus2 * y[4] - k2_bar * y[2] + (1 / p.Vc) * (J1 - J2 - p.vmaxkm * y[2])
        dCm = (1 / V3) * J2
        dCcBc = k2_bar * y[2] - kminus2 * y[4]
        
        return np.array([dCtub, dCe_bar, dCc, dCm, dCcBc])
    
    def simulate(self, time_points: np.ndarray, 
                params_opt: np.ndarray = None) -> np.ndarray:
        """
        Simulate PBPK model
        
        Args:
            time_points: Time points for simulation (min)
            params_opt: Optional optimizable parameters [k2_bar, kminus2, kd3, deltam, ps1]
            
        Returns:
            Array of shape (len(time_points), 5) with compartment concentrations
        """
        if params_opt is None:
            params_opt = np.array([
                self.params.k2_bar,
                self.params.kminus2,
                self.params.kd3,
                self.params.deltam,
                self.params.ps1
            ])
        
        # Initial conditions
        y0 = np.zeros(5)
        
        # Solve ODE
        solution = odeint(self.ode_system, y0, time_points, args=(params_opt,))
        
        return solution
    
    def get_vascular_concentration(self, solution: np.ndarray) -> np.ndarray:
        """Extract vascular concentration (observable)"""
        return solution[:, 1]

class PBPKOptimizer:
    """Parameter optimization for PBPK model"""
    
    def __init__(self, model: PBPKModel):
        self.model = model
        
    def objective_function(self, params_opt: np.ndarray,
                          time_points: np.ndarray,
                          observed_data: np.ndarray) -> np.ndarray:
        """
        Objective function for optimization (residuals)
        
        Args:
            params_opt: Parameter vector [k2_bar, kminus2, kd3, deltam, ps1]
            time_points: Experimental time points
            observed_data: Observed vascular concentrations
            
        Returns:
            Residuals (predicted - observed)
        """
        try:
            solution = self.model.simulate(time_points, params_opt)
            predicted = self.model.get_vascular_concentration(solution)
            residuals = predicted - observed_data
            return residuals
        except:
            return np.full_like(observed_data, 1e10)  # Large residual if simulation fails
    
    def fit_lsq(self, time_points: np.ndarray,
               observed_data: np.ndarray,
               p0: np.ndarray = None,
               bounds: Tuple = None) -> Dict:
        """
        Least-squares curve fitting
        
        Args:
            time_points: Experimental time points
            observed_data: Observed concentrations
            p0: Initial parameter guess
            bounds: Parameter bounds (lower, upper)
            
        Returns:
            Dictionary with optimization results
        """
        if p0 is None:
            p0 = np.array([
                self.model.params.k2_bar,
                self.model.params.kminus2,
                self.model.params.kd3,
                self.model.params.deltam,
                self.model.params.ps1
            ])
        
        if bounds is None:
            bounds = (
                [0.0001, 0.0001, 0.0001, -200, 0.001],  # Lower bounds
                [200, 300, 300, -50, 300]                # Upper bounds
            )
        
        print("Running Least-Squares Optimization...")
        
        result = least_squares(
            self.objective_function,
            p0,
            args=(time_points, observed_data),
            bounds=bounds,
            method='trf',  # Trust Region Reflective
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=1000,
            verbose=2
        )
        
        # Calculate statistics
        residuals = result.fun
        sse = np.sum(residuals**2)
        
        # Jacobian for uncertainty analysis
        jacobian = result.jac
        
        # Approximate covariance matrix
        try:
            cov = np.linalg.inv(jacobian.T @ jacobian)
            std_errors = np.sqrt(np.diag(cov))
        except:
            std_errors = np.full(len(result.x), np.nan)
            cov = None
        
        return {
            'params': result.x,
            'success': result.success,
            'sse': sse,
            'residuals': residuals,
            'jacobian': jacobian,
            'std_errors': std_errors,
            'covariance': cov,
            'nfev': result.nfev
        }
    
    def fit_monte_carlo(self, time_points: np.ndarray,
                       observed_data: np.ndarray,
                       n_iterations: int = 100,
                       p0: np.ndarray = None) -> Dict:
        """
        Monte Carlo parameter estimation
        
        Args:
            time_points: Experimental time points
            observed_data: Observed concentrations
            n_iterations: Number of Monte Carlo iterations
            p0: Central parameter values for sampling
            
        Returns:
            Dictionary with Monte Carlo results
        """
        if p0 is None:
            p0 = np.array([
                self.model.params.k2_bar,
                self.model.params.kminus2,
                self.model.params.kd3,
                self.model.params.deltam,
                self.model.params.ps1
            ])
        
        print(f"Running Monte Carlo Optimization ({n_iterations} iterations)...")
        
        all_params = []
        all_sse = []
        
        bounds = (
            [0.0001, 0.0001, 0.0001, -200, 0.001],
            [200, 300, 300, -50, 300]
        )
        
        for i in range(n_iterations):
            # Sample initial guess within ±30% of p0
            p_init = p0 * (1 + (np.random.rand(len(p0)) - 0.5) * 0.6)
            
            # Ensure bounds
            p_init = np.clip(p_init, bounds[0], bounds[1])
            
            try:
                result = least_squares(
                    self.objective_function,
                    p_init,
                    args=(time_points, observed_data),
                    bounds=bounds,
                    method='trf',
                    ftol=1e-6,
                    xtol=1e-6,
                    max_nfev=500,
                    verbose=0
                )
                
                if result.success:
                    all_params.append(result.x)
                    sse = np.sum(result.fun**2)
                    all_sse.append(sse)
                
            except:
                pass
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{n_iterations} iterations")
        
        if len(all_params) == 0:
            raise ValueError("No successful optimizations in Monte Carlo")
        
        all_params = np.array(all_params)
        all_sse = np.array(all_sse)
        
        # Find best fit
        best_idx = np.argmin(all_sse)
        best_params = all_params[best_idx]
        best_sse = all_sse[best_idx]
        
        # Calculate statistics
        mean_params = np.mean(all_params, axis=0)
        std_params = np.std(all_params, axis=0)
        
        # 95% confidence intervals (percentile method)
        ci_lower = np.percentile(all_params, 2.5, axis=0)
        ci_upper = np.percentile(all_params, 97.5, axis=0)
        
        return {
            'best_params': best_params,
            'best_sse': best_sse,
            'mean_params': mean_params,
            'std_params': std_params,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_params': all_params,
            'all_sse': all_sse,
            'n_successful': len(all_params)
        }

def load_experimental_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load experimental data from CSV"""
    data = pd.read_csv(filepath)
    time = data.iloc[:, 0].values
    concentration = data.iloc[:, 1].values
    return time, concentration

def visualize_results(time_points: np.ndarray,
                     observed_data: np.ndarray,
                     model: PBPKModel,
                     opt_params: np.ndarray,
                     method: str = "LSQ"):
    """Create comprehensive visualization of results"""
    
    # Simulate with optimized parameters
    solution = model.simulate(time_points, opt_params)
    predicted = model.get_vascular_concentration(solution)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Main fit plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_points, observed_data, 'ko', markersize=4, 
            label='Experimental Data', alpha=0.6)
    ax1.plot(time_points, predicted, 'b-', linewidth=2, 
            label='Model Fit')
    ax1.set_xlabel('Time (min)', fontsize=11)
    ax1.set_ylabel('Vascular Concentration (µM)', fontsize=11)
    ax1.set_title(f'Model Fit ({method})', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax2 = plt.subplot(2, 3, 2)
    residuals = predicted - observed_data
    ax2.plot(time_points, residuals, 'ro-', linewidth=1, markersize=3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time (min)', fontsize=11)
    ax2.set_ylabel('Residuals (µM)', fontsize=11)
    ax2.set_title('Residuals', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. All compartments
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_points, solution[:, 1], 'b-', linewidth=2, label='Vascular')
    ax3.plot(time_points, solution[:, 2], 'g-', linewidth=2, label='Cytoplasm (free)')
    ax3.plot(time_points, solution[:, 3], 'r-', linewidth=2, label='Mitochondria')
    ax3.plot(time_points, solution[:, 4], 'm-', linewidth=2, label='Cytoplasm (bound)')
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.set_ylabel('Concentration (µM)', fontsize=11)
    ax3.set_title('All Compartments', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax4 = plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter bar chart
    ax5 = plt.subplot(2, 3, 5)
    param_names = ['k2_bar', 'kminus2', 'kd3', 'deltam', 'ps1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax5.bar(param_names, opt_params, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Parameter Value', fontsize=11)
    ax5.set_title('Optimized Parameters', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    sse = np.sum(residuals**2)
    rmse = np.sqrt(np.mean(residuals**2))
    r_squared = 1 - (sse / np.sum((observed_data - np.mean(observed_data))**2))
    
    summary_text = f"""
    OPTIMIZATION RESULTS
    ══════════════════════════════
    
    Method: {method}
    
    Fit Statistics:
      SSE:        {sse:.6f}
      RMSE:       {rmse:.6f}
      R²:         {r_squared:.6f}
    
    Optimized Parameters:
      k2_bar:     {opt_params[0]:.4f}
      kminus2:    {opt_params[1]:.4f}
      kd3:        {opt_params[2]:.4f}
      deltam:     {opt_params[3]:.4f} mV
      ps1:        {opt_params[4]:.4f}
    
    Data Points: {len(time_points)}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('pbpk_optimization_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'pbpk_optimization_results.png'")
    plt.show()

def main():
    """Main execution function"""
    
    print("="*70)
    print("PBPK MODEL: R6G LUNG PERFUSION KINETICS")
    print("Parameter Optimization and Model Validation")
    print("="*70)
    
    # Initialize model with default parameters
    params = PBPKParameters()
    model = PBPKModel(params)
    optimizer = PBPKOptimizer(model)
    
    # Generate synthetic data (in real use, load from CSV)
    print("\n1. Generating synthetic experimental data...")
    time_points = np.linspace(0, 22, 220)
    true_params = np.array([6.661, 0.0689, 0.0124, -121.415, 53.59])
    true_solution = model.simulate(time_points, true_params)
    observed_data = model.get_vascular_concentration(true_solution)
    
    # Add noise to simulate experimental variability
    np.random.seed(42)
    observed_data += np.random.normal(0, 0.01 * np.max(observed_data), len(observed_data))
    
    print(f"   Time range: {time_points[0]:.1f} - {time_points[-1]:.1f} min")
    print(f"   Data points: {len(observed_data)}")
    
    # Method selection
    print("\n2. Select optimization method:")
    print("   [1] Least-Squares Curve Fitting")
    print("   [2] Monte Carlo Parameter Estimation")
    
    method_choice = input("   Enter choice (1 or 2): ").strip()
    
    if method_choice == "1":
        # Least-squares optimization
        print("\n3. Running Least-Squares Optimization...")
        
        p0 = true_params * (1 + (np.random.rand(5) - 0.5) * 0.2)  # ±10% initial guess
        
        result = optimizer.fit_lsq(time_points, observed_data, p0=p0)
        
        print("\n" + "="*70)
        print("LEAST-SQUARES RESULTS")
        print("="*70)
        print(f"Success: {result['success']}")
        print(f"Function evaluations: {result['nfev']}")
        print(f"SSE: {result['sse']:.6f}")
        
        param_names = ['k2_bar', 'kminus2', 'kd3', 'deltam', 'ps1']
        print("\nOptimized Parameters:")
        for i, name in enumerate(param_names):
            print(f"  {name:12s} = {result['params'][i]:10.4f} ± {result['std_errors'][i]:10.4f}")
        
        # Visualize
        visualize_results(time_points, observed_data, model, result['params'], "LSQ")
        
    elif method_choice == "2":
        # Monte Carlo optimization
        print("\n3. Running Monte Carlo Optimization...")
        
        result = optimizer.fit_monte_carlo(time_points, observed_data, n_iterations=50)
        
        print("\n" + "="*70)
        print("MONTE CARLO RESULTS")
        print("="*70)
        print(f"Successful iterations: {result['n_successful']}")
        print(f"Best SSE: {result['best_sse']:.6f}")
        print(f"Mean SSE: {np.mean(result['all_sse']):.6f}")
        
        param_names = ['k2_bar', 'kminus2', 'kd3', 'deltam', 'ps1']
        print("\nBest Parameters:")
        for i, name in enumerate(param_names):
            print(f"  {name:12s} = {result['best_params'][i]:10.4f}")
        
        print("\nMean ± Std:")
        for i, name in enumerate(param_names):
            print(f"  {name:12s} = {result['mean_params'][i]:10.4f} ± {result['std_params'][i]:10.4f}")
        
        print("\n95% Confidence Intervals:")
        for i, name in enumerate(param_names):
            width = result['ci_upper'][i] - result['ci_lower'][i]
            print(f"  {name:12s}: [{result['ci_lower'][i]:10.4f}, {result['ci_upper'][i]:10.4f}] (width: {width:.4f})")
        
        # Visualize
        visualize_results(time_points, observed_data, model, result['best_params'], "Monte Carlo")
        
        # Additional Monte Carlo visualizations
        plot_monte_carlo_distributions(result, param_names)
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

def plot_monte_carlo_distributions(result: Dict, param_names: List[str]):
    """Plot parameter distributions from Monte Carlo"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        data = result['all_params'][:, i]
        
        ax.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(result['mean_params'][i], color='red', linestyle='--', 
                  linewidth=2, label='Mean')
        ax.axvline(result['ci_lower'][i], color='green', linestyle='--', 
                  linewidth=1, label='95% CI')
        ax.axvline(result['ci_upper'][i], color='green', linestyle='--', linewidth=1)
        
        ax.set_xlabel(param_names[i], fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{param_names[i]} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # SSE distribution
    ax = axes[5]
    ax.hist(result['all_sse'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(result['best_sse'], color='red', linestyle='--', 
              linewidth=2, label='Best SSE')
    ax.axvline(np.mean(result['all_sse']), color='blue', linestyle='--', 
              linewidth=2, label='Mean SSE')
    ax.set_xlabel('SSE', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('SSE Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distributions.png', dpi=300, bbox_inches='tight')
    print("\nMonte Carlo distributions saved as 'monte_carlo_distributions.png'")
    plt.show()

if __name__ == "__main__":
    main()