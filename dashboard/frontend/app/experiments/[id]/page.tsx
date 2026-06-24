'use client';

import { useState, useEffect } from 'react';
import { use } from 'react';
import { StatCard, SourceBadge, StatusBadge } from '../../../src/components/StatCard';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

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

interface ExperimentDetail {
  id: string;
  status: string;
  final_score: number | null;
  timestamp: string | null;
  source?: string;
  details?: { kept?: number; discarded?: number; best_mae?: number; first_mae?: number };
  iterations?: Iteration[];
}

const STATUS_COLORS: Record<string, string> = {
  keep: '#22c55e', discard: '#ef4444', baseline: '#3b82f6', crash: '#eab308',
};

export default function ExperimentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`/api/experiments/${encodeURIComponent(id)}`);
        if (res.ok) setExperiment(await res.json());
        else setExperiment(null);
      } catch (e) {
        console.error('Failed to fetch:', e);
        setExperiment(null);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [id]);

  if (loading) {
    return <div className="flex items-center justify-center min-h-[400px]"><p className="text-slate-400">Loading...</p></div>;
  }

  if (!experiment) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-400 text-lg">Experiment not found</p>
        <a href="/experiments" className="text-blue-400 hover:underline mt-4 inline-block">← Back</a>
      </div>
    );
  }

  const iterations = experiment.iterations || [];
  const d = experiment.details || {};
  const kept = iterations.filter(i => i.status === 'keep');
  const discarded = iterations.filter(i => i.status === 'discard');

  const chartData = iterations.map(i => ({
    iteration: i.iteration,
    test_mae: i.test_mae,
    status: i.status,
  }));

  const bestMae = Math.min(...iterations.filter(i => i.test_mae != null).map(i => i.test_mae!), Infinity);
  const firstMae = iterations.length > 0 ? iterations[0].test_mae : null;
  const impr = firstMae && bestMae < Infinity ? ((firstMae - bestMae) / firstMae * 100).toFixed(1) : null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 flex-wrap">
        <a href="/experiments" className="text-blue-400 hover:underline text-sm">← Experiments</a>
        <h1 className="text-xl font-bold font-mono truncate max-w-[500px]">{experiment.id}</h1>
        <SourceBadge source={experiment.source} />
        <StatusBadge status={experiment.status} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <StatCard label="Baseline MAE" value={firstMae?.toFixed(2) ?? '—'} color="#6b7280" />
        <StatCard label="Best MAE" value={bestMae < Infinity ? bestMae.toFixed(2) : '—'} color="#22c55e" />
        <StatCard label="Improvement" value={impr ? `-${impr}%` : '—'} color="#22c55e" />
        <StatCard label="Iterations" value={iterations.length} />
        <StatCard label="Kept" value={kept.length} color="#22c55e" sub={`${(kept.length / Math.max(iterations.length - 1, 1) * 100).toFixed(0)}% success rate`} />
        <StatCard label="Discarded" value={discarded.length} color="#ef4444" />
      </div>

      {chartData.length > 1 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Test MAE (log scale)</h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="iteration" stroke="#94a3b8" label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
              <YAxis stroke="#94a3b8" scale="log" domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                       formatter={(value: number) => [value.toFixed(2), 'Test MAE']} />
              <Legend />
              <Line type="monotone" dataKey="test_mae" stroke="#f59e0b" strokeWidth={2.5}
                    dot={(props: any) => {
                      const { cx, cy, payload, key } = props;
                      return <circle key={key} cx={cx} cy={cy} r={6} fill={STATUS_COLORS[payload.status] || '#6b7280'} stroke="white" strokeWidth={1.5} />;
                    }}
                    name="Test MAE" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Iteration Log with Reasoning</h2>
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
                      style={{color: STATUS_COLORS[it.status] || '#e2e8f0'}}>
                    {it.test_mae?.toFixed(2) ?? '—'}
                  </td>
                  <td className="py-2 px-2 text-slate-300 text-xs max-w-[180px]">{it.description}</td>
                  <td className="py-2 px-2 text-slate-400 text-xs max-w-[240px] leading-relaxed">
                    {(it.hypothesis || '').slice(0, 250)}
                  </td>
                  <td className="py-2 px-2 text-slate-500 text-xs max-w-[220px] leading-relaxed">
                    {(it.mechanism || '').slice(0, 200)}
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
        <h2 className="text-lg font-semibold mb-3">Experiment Info</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div><span className="text-slate-400 block">Experiment</span><span className="text-white font-mono text-xs">{experiment.id}</span></div>
          <div><span className="text-slate-400 block">Source</span><span className="text-white">{experiment.source || '—'}</span></div>
          <div><span className="text-slate-400 block">Status</span><span className="text-white">{experiment.status}</span></div>
          <div><span className="text-slate-400 block">Iterations</span><span className="text-white">{iterations.length}</span></div>
          <div><span className="text-slate-400 block">Kept</span><span className="text-emerald-400">{kept.length}</span></div>
          <div><span className="text-slate-400 block">Discarded</span><span className="text-red-400">{discarded.length}</span></div>
          <div><span className="text-slate-400 block">Best MAE</span><span className="text-emerald-400 font-bold">{bestMae < Infinity ? bestMae.toFixed(2) : '—'}</span></div>
          <div><span className="text-slate-400 block">Improvement</span><span className="text-emerald-400">{impr ? `-${impr}%` : '—'}</span></div>
        </div>
      </div>
    </div>
  );
}
