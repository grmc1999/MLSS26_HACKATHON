'use client';

import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { StatCard, ScoreChart } from '../src/components/StatCard';

interface Iteration {
  iteration: number;
  commit: string;
  test_mae: number | null;
  val_mae: number | null;
  status: string;
  description: string;
  hypothesis: string;
  mechanism: string;
  expected_delta: string;
  risk: string;
  change_type: string;
  diff_summary: string;
}

interface Experiment {
  id: string;
  final_score: number | null;
  source?: string;
  iterations?: Iteration[];
  details?: { best_mae?: number; first_mae?: number; kept?: number; discarded?: number };
}

export default function Home() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/experiments?task=flu');
        const data = await res.json();
        setExperiments(data.experiments || []);
      } catch (e) {
        console.error('Failed to fetch:', e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const loopExps = useMemo(() =>
    experiments.filter(e => e.source === 'auto_loop'),
    [experiments]
  );

  const activeExp = loopExps[0];
  const iterations = activeExp?.iterations || [];
  const details = activeExp?.details || {};

  const statusColors: Record<string, string> = {
    keep: '#22c55e', discard: '#ef4444', baseline: '#6b7280', crash: '#eab308',
  };

  const chartData = iterations.map(i => ({
    iteration: i.iteration,
    test_mae: i.test_mae,
    status: i.status,
  }));

  const bestMae = Math.min(...iterations.filter(i => i.test_mae != null).map(i => i.test_mae!), Infinity);
  const firstMae = iterations.length > 0 ? iterations[0].test_mae : null;
  const impr = firstMae && bestMae < Infinity ? ((firstMae - bestMae) / firstMae * 100).toFixed(1) : null;

  const kept = iterations.filter(i => i.status === 'keep').length;
  const discarded = iterations.filter(i => i.status === 'discard').length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">🤒 Flu Forecasting Dashboard</h1>
        <span className="text-sm text-slate-400">{loopExps.length} experiment(s)</span>
      </div>

      {!activeExp ? (
        <div className="bg-slate-800 rounded-lg p-8 text-center">
          <p className="text-slate-400">No auto-loop experiments found. Run the pipeline first.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <StatCard label="Baseline MAE" value={firstMae?.toFixed(2) ?? '—'} color="#6b7280" />
            <StatCard label="Best MAE" value={bestMae < Infinity ? bestMae.toFixed(2) : '—'} color="#22c55e" sub="lower is better" />
            <StatCard label="Improvement" value={impr ? `-${impr}%` : '—'} color="#22c55e" />
            <StatCard label="Iterations" value={iterations.length} />
            <StatCard label="Kept" value={kept} color="#22c55e" />
            <StatCard label="Discarded" value={discarded} color="#ef4444" />
          </div>

          {chartData.length > 1 && (
            <div className="bg-slate-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Test MAE Over Iterations (log scale)</h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="iteration" stroke="#94a3b8" label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
                  <YAxis stroke="#94a3b8" scale="log" domain={['auto', 'auto']} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                           formatter={(value: number) => [value.toFixed(2), 'Test MAE']} />
                  <Line type="monotone" dataKey="test_mae" stroke="#f59e0b" strokeWidth={2.5}
                        dot={(props: any) => {
                          const { cx, cy, payload, key } = props;
                          const color = statusColors[payload.status] || '#6b7280';
                          return <circle key={key} cx={cx} cy={cy} r={6} fill={color} stroke="white" strokeWidth={1.5} />;
                        }}
                        name="Test MAE" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Iteration Details with Research Reasoning</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700 bg-slate-900/50">
                    <th className="text-left py-2 px-2">#</th>
                    <th className="text-left py-2 px-2">Status</th>
                    <th className="text-right py-2 px-2">Test MAE</th>
                    <th className="text-left py-2 px-2" style={{minWidth:160}}>Change</th>
                    <th className="text-left py-2 px-2" style={{minWidth:220}}>Hypothesis</th>
                    <th className="text-left py-2 px-2" style={{minWidth:200}}>Mechanism</th>
                    <th className="text-center py-2 px-2">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {iterations.map((it) => (
                    <tr key={it.iteration} className="border-b border-slate-800 hover:bg-slate-700/30">
                      <td className="py-2 px-2 font-mono text-slate-400">{it.iteration}</td>
                      <td className="py-2 px-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          it.status === 'keep' ? 'bg-emerald-900/50 text-emerald-300' :
                          it.status === 'discard' ? 'bg-red-900/50 text-red-300' :
                          'bg-blue-900/50 text-blue-300'
                        }`}>{it.status}</span>
                      </td>
                      <td className="py-2 px-2 text-right font-mono font-bold"
                          style={{color: statusColors[it.status] || '#e2e8f0'}}>
                        {it.test_mae?.toFixed(2) ?? '—'}
                      </td>
                      <td className="py-2 px-2 text-slate-300 text-xs max-w-[180px]">{it.description}</td>
                      <td className="py-2 px-2 text-slate-400 text-xs max-w-[240px] leading-relaxed">
                        {(it.hypothesis || '').slice(0, 200)}
                      </td>
                      <td className="py-2 px-2 text-slate-500 text-xs max-w-[220px] leading-relaxed">
                        {(it.mechanism || '').slice(0, 180)}
                      </td>
                      <td className="py-2 px-2 text-center text-xs">
                        <span className={`${
                          it.risk === 'low' ? 'text-emerald-400' :
                          it.risk === 'medium' ? 'text-yellow-400' :
                          it.risk === 'high' ? 'text-red-400' : 'text-slate-400'
                        }`}>{it.risk || '—'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-slate-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3">Experiment Summary</h2>
            <p className="text-sm text-slate-300 leading-relaxed">
              This experiment used a conditional diffusion model (DiffusionForecaster + ConditionalDenoiser)
              for cross-country ILI forecasting, trained on US CDC data and tested on WHO FluID data
              (France, Mexico, Australia, South Africa). The pipeline ran <strong>{iterations.length - 1}</strong> iterations
              with RAG-guided modifications to <code className="text-blue-400">env/train.py</code>.
            </p>
            <p className="text-sm text-slate-300 mt-2">
              <strong className="text-emerald-400">Best result:</strong> Test MAE <strong>{bestMae < Infinity ? bestMae.toFixed(2) : '—'}</strong>
              {impr ? ` (${impr}% improvement from baseline ${firstMae?.toFixed(2)})` : ''}.
            </p>
          </div>
        </>
      )}

      <div className="text-center text-xs text-slate-500">
        <code>dashboard/backend/main.py</code> + <code>dashboard/frontend/</code>
      </div>
    </div>
  );
}
