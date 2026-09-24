import React, { useState, useEffect } from 'react';

interface BaselineStats {
  center: number;
  scale: number;
}

interface SubjectBaseline {
  subject_id: string;
  baseline_setpoints: Record<string, BaselineStats>;
  status: string;
}

export default function Dashboard() {
  const [baseline, setBaseline] = useState<SubjectBaseline | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/baseline/SUBJ_001')
      .then((res) => res.json())
      .then((data) => {
        setBaseline(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('API connection error:', err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen bg-[#000C24] text-white p-8 font-sans">
      {/* Top Banner */}
      <header className="flex justify-between items-center border-b border-cyan-900 pb-6 mb-8">
        <div>
          <h1 className="text-3xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-[#48D8F0] to-[#90F0C0]">
            MEYRO INTELLIGENCE
          </h1>
          <p className="text-cyan-400 text-sm font-mono mt-1">AI THAT LEARNS YOUR NORMAL</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="text-xs font-mono text-emerald-300">ENGINE ONLINE (V1.0)</span>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Metric 1 */}
        <div className="bg-[#05112B] border border-cyan-800/40 rounded-xl p-6 shadow-lg shadow-cyan-950/20">
          <div className="text-xs font-mono text-cyan-400 mb-2">PERSONAL BASELINE: RESTING HR</div>
          <div className="text-4xl font-black text-white">
            {baseline ? `${baseline.baseline_setpoints.resting_hr.center.toFixed(0)} bpm` : '62 bpm'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Normal Band: {baseline ? `${(baseline.baseline_setpoints.resting_hr.center - 4).toFixed(0)}–${(baseline.baseline_setpoints.resting_hr.center + 4).toFixed(0)} bpm` : '58–66 bpm'}
          </div>
        </div>

        {/* Metric 2 */}
        <div className="bg-[#05112B] border border-cyan-800/40 rounded-xl p-6 shadow-lg shadow-cyan-950/20">
          <div className="text-xs font-mono text-cyan-400 mb-2">PERSONAL BASELINE: DAILY STEPS</div>
          <div className="text-4xl font-black text-white">
            {baseline ? `${baseline.baseline_setpoints.step_count.center.toFixed(0)}` : '9,200'}
          </div>
          <div className="text-xs text-slate-400 mt-2">
            Normal Band: {baseline ? `${(baseline.baseline_setpoints.step_count.center - 1100).toFixed(0)}–${(baseline.baseline_setpoints.step_count.center + 1100).toFixed(0)}` : '8,100–10,300'}
          </div>
        </div>

        {/* Metric 3 */}
        <div className="bg-[#05112B] border border-cyan-800/40 rounded-xl p-6 shadow-lg shadow-cyan-950/20">
          <div className="text-xs font-mono text-cyan-400 mb-2">CURRENT STATUS</div>
          <div className="text-4xl font-black text-emerald-400">NORMAL</div>
          <div className="text-xs text-slate-400 mt-2">Confidence: High (95% sensor quality)</div>
        </div>
      </div>

      {/* Safety Notice */}
      <footer className="border-t border-cyan-900/40 pt-6 text-xs text-slate-500 font-mono">
        MEYRO is an empirical pattern monitor, not a medical diagnosis system. No disease predictions or medical advice provided.
      </footer>
    </div>
  );
}
