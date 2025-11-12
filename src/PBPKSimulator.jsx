
import html2canvas from 'html2canvas';
import React, { useState, useRef, useEffect } from 'react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, TrendingUp, Settings, FileUp, PlayCircle, Database } from 'lucide-react';

const PBPKSimulator = () => {
  const chartRef = useRef(null);  
  const [params, setParams] = useState({
    k2_bar: 6.661,
    kminus2: 0.0689,
    kd3: 0.0124,
    deltam: -121.415,
    ps1: 53.59
  });
  
  const [optimizationMethod, setOptimizationMethod] = useState('lsqcurvefit');
  const [dataType, setDataType] = useState('normoxia');
  const [uploadedData, setUploadedData] = useState(null);
  const [simulationData, setSimulationData] = useState([]);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [showCompartments, setShowCompartments] = useState(false);

  // PBPK Model ODE solver (simplified Euler method)
  const solvePBPK = (params, timePoints) => {
    const { k2_bar, kminus2, kd3, deltam, ps1 } = params;
    
    // Fixed parameters
    const Ve = 0.85, Vc = 1, Vm = 0.02 * Vc, Vtub = 4;
    const alpha = 0.0374158, Be = 0.5, kd1 = 0.32, ps2 = 1.123;
    const F = 10, vmaxkm = 0, deltap = -43, deltam_un = -0.1;
    
    const V3 = Vm * (1 + (1 / kd3));
    const dt = 0.01; // time step
    
    let results = [];
    let state = [0, 0, 0, 0, 0]; // [Ctub, Ce_bar, Cc, Cm, CcBc]
    
    for (let t of timePoints) {
      let currentT = results.length > 0 ? results[results.length - 1].time : 0;
      
      while (currentT < t) {
        // Determine Cin based on phase
        let Cin = 0;
        let currentDeltam = deltam;
        
        if (currentT >= 0 && currentT <= 10) {
          Cin = 0.25;
        } else if (currentT > 10 && currentT < 15.5) {
          Cin = 0;
        } else if (currentT >= 15.5) {
          Cin = 0;
          currentDeltam = deltam_un;
        }
        
        // Calculate Ce (free concentration)
        const Ce = state[1] / (1 + Be / kd1);
        
        // Calculate fluxes
        const J1 = -((alpha * ps1 * deltap) / (Math.exp(-alpha * deltap) - 1)) * 
                   (Math.exp(-alpha * deltap) * Ce - state[2]);
        const J2 = -((alpha * ps2 * currentDeltam) / (Math.exp(-alpha * currentDeltam) - 1)) * 
                   (Math.exp(-alpha * currentDeltam) * state[2] - state[3]);
        
        // ODEs
        const dCtub = (1.0 / Vtub) * F * (Cin - state[0]);
        const dCe_bar = (1 / Ve) * (-J1 + vmaxkm * state[2] + F * (state[0] - state[1]));
        const dCc = kminus2 * state[4] - k2_bar * state[2] + (1 / Vc) * (J1 - J2 - vmaxkm * state[2]);
        const dCm = (1 / V3) * J2;
        const dCcBc = k2_bar * state[2] - kminus2 * state[4];
        
        // Euler integration
        state[0] += dCtub * dt;
        state[1] += dCe_bar * dt;
        state[2] += dCc * dt;
        state[3] += dCm * dt;
        state[4] += dCcBc * dt;
        
        currentT += dt;
      }
      
      results.push({
        time: t,
        Ctub: state[0],
        Ce_bar: state[1],
        Cc: state[2],
        Cm: state[3],
        CcBc: state[4]
      });
    }
    
    return results;
  };

  // Generate synthetic experimental data
  const generateExperimentalData = (dataType) => {
    const timePoints = Array.from({ length: 220 }, (_, i) => i * 0.1);
    
    // True parameters for data generation
    const trueParams = dataType === 'normoxia' 
      ? { k2_bar: 6.661, kminus2: 0.0689, kd3: 0.0124, deltam: -121.415, ps1: 53.59 }
      : { k2_bar: 7.63, kminus2: 0.0699, kd3: 0.0108, deltam: -116.415, ps1: 53.59 };
    
    const results = solvePBPK(trueParams, timePoints);
    
    // Add noise to simulate experimental data
    return results.map(r => ({
      time: r.time,
      concentration: r.Ce_bar + (Math.random() - 0.5) * 0.02 * r.Ce_bar
    }));
  };

   // === Least Squares Curve Fitting (Improved) ===
   const lsqCurveFit = (experimentalData) => {
      const timePoints = experimentalData.map(d => d.time);
      const expConc = experimentalData.map(d => d.concentration);
      let bestParams = { ...params };
      let bestSSE = Infinity;
   
      const paramKeys = Object.keys(bestParams);
      const learningRate = 0.05;  // larger step for visible optimization
      const iterations = 150;
   
      for (let iter = 0; iter < iterations; iter++) {
      const predicted = solvePBPK(bestParams, timePoints);
      const predConc = predicted.map(p => p.Ce_bar);
      let sse = 0;
      for (let i = 0; i < expConc.length; i++) {
         sse += Math.pow(expConc[i] - predConc[i], 2);
      }
   
      if (sse < bestSSE) {
         bestSSE = sse;
      }
   
      // Random small perturbation per iteration
      const randomParam = paramKeys[Math.floor(Math.random() * paramKeys.length)];
      const perturbation = (Math.random() - 0.5) * learningRate * Math.abs(bestParams[randomParam]);
      bestParams[randomParam] += perturbation;
   
      // keep params within physiological bounds
      bestParams.k2_bar = Math.max(0.1, Math.min(50, bestParams.k2_bar));
      bestParams.kminus2 = Math.max(0.001, Math.min(1, bestParams.kminus2));
      bestParams.kd3 = Math.max(0.001, Math.min(0.1, bestParams.kd3));
      bestParams.deltam = Math.max(-200, Math.min(-50, bestParams.deltam));
      bestParams.ps1 = Math.max(1, Math.min(100, bestParams.ps1));
      }
   
      return { params: bestParams, sse: bestSSE };
   };
   
   
   // === Monte Carlo Optimization (Fixed) ===
   const monteCarloOptimization = (experimentalData, nIterations = 40) => {
      const timePoints = experimentalData.map(d => d.time);
      const expConc = experimentalData.map(d => d.concentration);
      const allResults = [];
   
      for (let i = 0; i < nIterations; i++) {
      // Start from random params around the base
      const trialParams = {
         k2_bar: params.k2_bar * (1 + (Math.random() * 0.5 - 0.25)),
         kminus2: params.kminus2 * (1 + (Math.random() * 0.5 - 0.25)),
         kd3: params.kd3 * (1 + (Math.random() * 0.5 - 0.25)),
         deltam: params.deltam * (1 + (Math.random() * 0.4 - 0.2)),
         ps1: params.ps1 * (1 + (Math.random() * 0.4 - 0.2))
      };
   
      const sim = solvePBPK(trialParams, timePoints);
      const predConc = sim.map(p => p.Ce_bar);
      let sse = 0;
      for (let i = 0; i < expConc.length; i++) {
         sse += Math.pow(expConc[i] - predConc[i], 2);
      }
   
      allResults.push({ params: trialParams, sse });
      }
   
      // Pick best
      allResults.sort((a, b) => a.sse - b.sse);
      const bestResult = allResults[0];
   
      // Stats
      const paramStats = {};
      Object.keys(params).forEach(key => {
      const values = allResults.map(r => r.params[key]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(a - mean, 2), 0) / values.length);
      paramStats[key] = { mean, std };
      });
   
      return {
      bestParams: bestResult.params,
      bestSSE: bestResult.sse,
      allResults,
      statistics: paramStats
      };
   };
   
   
   const runSimulation = () => {
      const timePoints = Array.from({ length: 220 }, (_, i) => i * 0.1);
      const results = solvePBPK(params, timePoints);
      setSimulationData(results);
    };

   // === Optimization Runner (Updated) ===
   const runOptimization = () => {
      setIsRunning(true);
   
      setTimeout(() => {
      const expData = uploadedData && uploadedData.length > 0
         ? uploadedData
         : generateExperimentalData(dataType);
   
      let results;
      if (optimizationMethod === 'lsqcurvefit') {
         results = lsqCurveFit(expData);
         results.experimentalData = expData;
      } else {
         results = monteCarloOptimization(expData, 40);
         results.experimentalData = expData;
      }
   
      setOptimizationResults(results);
      setParams(optimizationMethod === 'lsqcurvefit' ? results.params : results.bestParams);
      setIsRunning(false);
      }, 500);
   };
   
   
   // === Data for plotting (smooth fit) ===
   const getMainPlotData = () => {
      if (!simulationData || simulationData.length === 0) return [];
   
      const sim = simulationData;
      const exp = optimizationResults?.experimentalData || [];
   
      // merge by nearest time for smoother display
      return sim.map(s => {
      const nearestExp = exp.reduce((prev, curr) =>
         Math.abs(curr.time - s.time) < Math.abs(prev.time - s.time) ? curr : prev,
         exp[0] || { concentration: 0 }
      );
      return {
         time: s.time,
         model: s.Ce_bar,
         experimental: nearestExp.concentration
      };
      });
   };
 
   
   const handleFileUpload = (e) => {
   const file = e.target.files[0];
   if (!file) return;

   const reader = new FileReader();
   reader.onload = (event) => {
      const csv = event.target.result;
      const rows = csv.split("\n").slice(1);
      const parsed = rows
         .map((row) => {
         const [time, concentration] = row.split(",").map(Number);
         if (!isNaN(time) && !isNaN(concentration)) {
            return { time, concentration };
         }
         return null;
         })
         .filter((r) => r !== null);

      setUploadedData(parsed);
      console.log("✅ Uploaded experimental data loaded:", parsed);
   };
   reader.readAsText(file);
   };

  useEffect(() => {
   if (!uploadedData) {
     const syntheticData = generateExperimentalData(dataType);
     setOptimizationResults({ experimentalData: syntheticData });
     runSimulation();
   }
 }, [dataType, params]);

  const getCompartmentData = () => {
    return simulationData.filter((_, idx) => idx % 10 === 0).map(d => ({
      time: d.time,
      Vascular: d.Ce_bar,
      Cytoplasm: d.Cc,
      Mitochondria: d.Cm,
      Bound: d.CcBc
    }));
  };

  const getParameterDistribution = () => {
    if (!optimizationResults?.allResults) return [];
    
    const allParams = optimizationResults.allResults;
    return Object.keys(params).map(key => ({
      parameter: key,
      mean: optimizationResults.statistics[key].mean,
      std: optimizationResults.statistics[key].std
    }));
  };
  
  const downloadChart = () => {
   if (!chartRef.current) return;

   html2canvas(chartRef.current, { backgroundColor: null }).then((canvas) => {
     const link = document.createElement('a');
     link.download = 'PBPK_Simulation.png';
     link.href = canvas.toDataURL('image/png');
     link.click();
   });
 };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Activity className="w-10 h-10" />
            PBPK Model: R6G Lung Perfusion Kinetics
          </h1>
          <p className="text-blue-200 text-lg">Physiologically-Based Pharmacokinetic Modeling & Parameter Optimization</p>
        </div>

        {/* Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Parameters */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Model Parameters
            </h2>
            
            {Object.entries(params).map(([key, value]) => (
              <div key={key} className="mb-3">
                <label className="text-blue-200 text-sm block mb-1">
                  {key}: {value.toFixed(4)}
                </label>
                <input
                  type="range"
                  min={key === 'deltam' ? -200 : 0.001}
                  max={key === 'deltam' ? -50 : key === 'ps1' ? 100 : key === 'k2_bar' ? 50 : 5}
                  step={key === 'deltam' ? 1 : 0.001}
                  value={value}
                  onChange={(e) => setParams({...params, [key]: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>
            ))}
          </div>

          {/* Optimization Settings */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Optimization Settings
            </h2>
            
            <div className="mb-4">
              <label className="text-blue-200 text-sm block mb-2">Data Type</label>
              <select
                value={dataType}
                onChange={(e) => setDataType(e.target.value)}
                className="w-full bg-slate-800 text-white p-2 rounded border border-white/20"
              >
                <option value="normoxia">Normoxia</option>
                <option value="hyperoxia">Hyperoxia</option>
              </select>
            </div>

            <div className="mb-4">
            <label className="text-sm block mb-2 text-blue-200">Upload Experimental Data (CSV)</label>
            <div className="flex items-center gap-2">
              <FileUp className="w-5 h-5 text-blue-400" />
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="w-full text-sm bg-slate-800 text-white p-1 rounded border border-white/20 cursor-pointer"
              />
            </div>
            {uploadedData && (
              <p className="text-green-400 text-xs mt-2">
                Loaded {uploadedData.length} data points
              </p>
            )}
          </div>

            <div className="mb-4">
              <label className="text-blue-200 text-sm block mb-2">Optimization Method</label>
              <select
                value={optimizationMethod}
                onChange={(e) => setOptimizationMethod(e.target.value)}
                className="w-full bg-slate-800 text-white p-2 rounded border border-white/20"
              >
                <option value="lsqcurvefit">Least Squares Curve Fit</option>
                <option value="montecarlo">Monte Carlo</option>
              </select>
            </div>

            <button
              onClick={runOptimization}
              disabled={isRunning}
              className="w-full bg-gradient-to-r from-green-500 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-green-600 hover:to-blue-700 transition-all disabled:opacity-50 flex items-center justify-center gap-2 mb-3"
            >
              <PlayCircle className="w-5 h-5" />
              {isRunning ? 'Optimizing...' : 'Run Optimization'}
            </button>

            <button
              onClick={runSimulation}
              className="w-full bg-white/20 text-white py-2 rounded-lg font-semibold hover:bg-white/30 transition-all border border-white/30"
            >
              Update Simulation
            </button>

            <div className="mt-4">
              <label className="flex items-center text-white cursor-pointer">
                <input
                  type="checkbox"
                  checked={showCompartments}
                  onChange={(e) => setShowCompartments(e.target.checked)}
                  className="mr-2"
                />
                Show All Compartments
              </label>
            </div>
          </div>

          {/* Results Summary */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Results Summary
            </h2>
            <p className="text-blue-300 text-xs mt-2">
               Using {uploadedData ? 'uploaded experimental data' : 'synthetic data'} for optimization.
            </p>

            
            {optimizationResults ? (
            <div className="space-y-3">
               <div className="bg-green-500/20 p-3 rounded border border-green-500/30">
                  <div className="text-green-200 text-xs mb-1">Sum of Squared Errors</div>
                  <div className="text-2xl font-bold text-white">
                  {(optimizationResults.sse || optimizationResults.bestSSE)?.toFixed(6)}
                  </div>
               </div>

               {optimizationMethod === 'montecarlo' && (
                  <div className="bg-blue-500/20 p-3 rounded border border-blue-500/30">
                  <div className="text-blue-200 text-xs mb-1">Iterations Completed</div>
                  <div className="text-2xl font-bold text-white">
                     {optimizationResults.allResults?.length || 0}
                  </div>
                  </div>
               )}

               <div className="text-sm text-blue-200">
                  <div className="font-semibold mb-2">Optimized Parameters:</div>
                  {Object.entries(
                  optimizationMethod === 'lsqcurvefit'
                     ? optimizationResults.params || {}      // fallback for LSQ
                     : optimizationResults.bestParams || {} // fallback for Monte Carlo
                  ).map(([key, val]) => (
                  <div key={key} className="flex justify-between py-1">
                     <span>{key}:</span>
                     <span className="font-mono">{val?.toFixed(4)}</span> {/* safe optional chaining */}
                  </div>
                  ))}
               </div>
            </div>
            ) : (
            <div className="text-center text-blue-300 py-8">
               Run optimization to see results
            </div>
            )}

          </div>
        </div>

        {/* Main Visualization */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
          <h3 className="text-lg font-bold text-white mb-4">
            {optimizationResults ? 'Model Fit vs Experimental Data' : 'PBPK Model Simulation'}
          </h3>
          <button
            onClick={downloadChart}
            className="bg-green-500 text-white px-4 py-2 rounded mb-4"
          >
            Download Graph
         </button>
         <div ref={chartRef}>
         <ResponsiveContainer width="100%" height={400}>
            <LineChart data={getMainPlotData()} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
               
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis 
               dataKey="time"
               type="category"       // <- important
               ticks={[
                  0, 0.666666667, 1.333333333, 2, 3, 4, 5, 7, 9, 10, 11, 12,
                  13, 14, 15, 15.33333333, 15.66666667, 16, 16.33333333, 16.66666667,
                  17, 18, 19, 20, 21, 22
               ]}
               label={{ value: 'Time (min)', position: 'insideBottom', offset: -30, fill: '#60a5fa' }}
               stroke="#60a5fa"
               />
              <YAxis 
                label={{ value: 'Concentration (µM)', angle: -90, position: 'insideLeft', fill: '#60a5fa' }}
                stroke="#60a5fa"
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }}
              />
              <Legend />
              {optimizationResults?.experimentalData && (
                <Scatter 
                  data={getMainPlotData()} 
                  dataKey="experimental"
                  name="Experimental Data"
                  fill="#ef4444"
                  line={false}
                  shape="circle"
                />
              )}
              <Line 
                type="monotone" 
                dataKey="model" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Model Prediction"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
         </div>
        </div>

        {/* Compartment Visualization */}
        {showCompartments && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Compartment Concentrations</h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={getCompartmentData()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="time" 
                  label={{ value: 'Time (min)', position: 'insideBottom', offset: -10, fill: '#60a5fa' }}
                  stroke="#60a5fa"
                />
                <YAxis 
                  label={{ value: 'Concentration (µM)', angle: -90, position: 'insideLeft', fill: '#60a5fa' }}
                  stroke="#60a5fa"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }}
                />
                <Legend />
                <Line type="monotone" dataKey="Vascular" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Cytoplasm" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Mitochondria" stroke="#f59e0b" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Bound" stroke="#ec4899" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Parameter Distribution (Monte Carlo only) */}
        {optimizationMethod === 'montecarlo' && optimizationResults?.statistics && (
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <h3 className="text-lg font-bold text-white mb-4">Parameter Distribution (Monte Carlo)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={getParameterDistribution()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis 
                  dataKey="parameter" 
                  stroke="#60a5fa"
                />
                <YAxis 
                  stroke="#60a5fa"
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6' }}
                />
                <Legend />
                <Bar dataKey="mean" fill="#3b82f6" name="Mean Value" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Info Panel */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
          <h3 className="text-lg font-bold text-white mb-3">About This Model</h3>
          <p className="text-blue-200 leading-relaxed">
            This PBPK model simulates rhodamine 6G (R6G) kinetics in lung perfusion experiments. The model includes 
            five compartments: tubing, vascular, cytoplasmic (free and bound), and mitochondrial regions. Parameter 
            optimization uses least-squares curve fitting or Monte Carlo methods to fit experimental time-concentration 
            data. The model incorporates membrane potentials, protein binding, and multi-phase experimental protocols 
            (loading, wash, uncoupler phases).
          </p>
        </div>
      </div>
    </div>
  );
};

export default PBPKSimulator;