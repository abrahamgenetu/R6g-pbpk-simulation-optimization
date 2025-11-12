
# PBPK Simulator: R6G Lung Perfusion Kinetics

A **Physiologically-Based Pharmacokinetic (PBPK) simulator** for rhodamine 6G (R6G) kinetics in lung perfusion experiments. This interactive web application allows users to:

* Adjust **PBPK model parameters** using sliders.
* Run **simulations** in real-time to observe concentration-time profiles.
* Perform **parameter optimization** using **Least Squares Curve Fit** or **Monte Carlo** methods.
* Compare simulation results to  **experimental or uploaded data** .
* Download generated graphs for reporting or analysis.

  ![1762971133661](image/README/1762971133661.png)

---

## Table of Contents

* [Features]()
* [Installation]()
* [Usage]()
* [Project Structure]()
* [PBPK Model Details]()
* [Optimization Methods]()
* [Future Improvements]()

---

## Features

1. **Interactive Parameter Control**
   * Adjust model parameters such as `k2_bar`, `kminus2`, `kd3`, `deltam`, and `ps1` in real-time.
   * Run simulations directly from the left-side control panel.
2. **Simulation Visualization**
   * Display model predictions as a smooth line graph.
   * Compare with experimental data using scatter points at exact times.
3. **Parameter Optimization**
   * Least Squares Curve Fit (LSQ)
   * Monte Carlo Optimization
   * Provides  **Sum of Squared Errors (SSE)** , parameter statistics, and iteration counts.
4. **Data Upload**
   * Upload experimental CSV data for comparison (time vs concentration).
   * Supports synthetic data generation if no upload is provided.
5. **Downloadable Graphs**
   * Export PBPK simulation plots as PNG images for reports or presentations.
6. **Compartment Visualization**
   * Display individual compartment concentrations: Vascular, Cytoplasm, Mitochondria, Bound.

---

## Installation

1. **Clone the repository:**

<pre class="overflow-visible!" data-start="2140" data-end="2230"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>git </span><span>clone</span><span> https://github.com/yourusername/pbpk-simulator.git
</span><span>cd</span><span> pbpk-simulator
</span></span></code></div></div></pre>

2. **Install dependencies:**

<pre class="overflow-visible!" data-start="2261" data-end="2284"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>npm install
</span></span></code></div></div></pre>

3. **Run the application:**

<pre class="overflow-visible!" data-start="2314" data-end="2335"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>npm start
</span></span></code></div></div></pre>

The app will run on `http://localhost:3000`.

---

## Usage

1. **Adjust Parameters:**
   * Use sliders in the left panel to adjust PBPK parameters.
2. **Run Simulation:**
   * Click **Run Simulation** to generate model predictions.
3. **Run Optimization (Optional):**
   * Select optimization method (LSQ or Monte Carlo).
   * Click **Run Optimization** to fit the model to experimental data.
4. **Upload Experimental Data (Optional):**
   * Upload a CSV file with columns `time,concentration` for comparison.
5. **Download Graphs:**
   * Click **Download Graph** to save the current plot as a PNG image.
6. **Show Compartments:**
   * Toggle **Show All Compartments** to view individual compartment concentrations.

---

## Project Structure

<pre class="overflow-visible!" data-start="3113" data-end="3406"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>src/
 ├─ components/
 │   └─ PBPKSimulator.jsx       </span><span># Main component</span><span>
 ├─ App.jsx                     </span><span># App entry point</span><span>
 ├─ index.js                    </span><span># React DOM render</span><span>
 ├─ utils/
 │   └─ pbpkModel.js            </span><span># PBPK solver & optimization functions</span><span>
 └─ styles/
     └─ tailwind.css
</span></span></code></div></div></pre>

---

## PBPK Model Details

* **Compartments:** Tubing, Vascular, Cytoplasm, Mitochondria, Bound.
* **ODE Solver:** Simplified Euler integration method.
* **Features:**
  * Multi-phase experimental protocols (loading, wash, uncoupler).
  * Membrane potentials, protein binding, and metabolite fluxes.

---

## Optimization Methods

1. **Least Squares Curve Fit (LSQ):**
   * Minimizes sum of squared errors between experimental and model data.
2. **Monte Carlo Optimization:**
   * Randomly samples parameter space.
   * Returns best parameters, SSE, and parameter statistics (mean & std).

---

## Future Improvements

* Add **real-time animation** of compartment changes.
* Integrate **more advanced ODE solvers** (e.g., Runge-Kutta).
* Include **sensitivity analysis** for parameters.
* Allow **multiple experimental datasets** comparison.

---

## License

MIT License © 2025 Abraham Taye
