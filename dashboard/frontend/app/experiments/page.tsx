'use client';

import { useState, useEffect, useMemo } from 'react';
import { SourceBadge, StatusBadge, getExpColor } from '../../src/components/StatCard';

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  total_steps?: number;
  timestamp: string | null;
  source?: string;
  details?: Record<string, unknown>;
  runtime?: string;
  steps?: { step: number }[];
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        setExperiments(data.experiments || []);
      } catch (e) {
        console.error('Failed to fetch experiments:', e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const filtered = useMemo(() => {
    if (filter === 'all') return experiments;
    if (filter === 'loops') return experiments.filter(e => e.source === 'auto_loop');
    return experiments.filter(e => e.status === filter);
  }, [experiments, filter]);

  const stats = useMemo(() => ({
    total: experiments.length,
    loops: experiments.filter(e => e.source === 'auto_loop').length,
    run_exp: experiments.filter(e => e.source === 'run_exp').length,
    logs: experiments.filter(e => !e.source || e.source === 'logs').length,
  }), [experiments]);

  const filters = [
    { key: 'all', label: 'All', count: stats.total },
    { key: 'loops', label: 'Loops', count: stats.loops },
    { key: 'completed', label: 'Completed', count: experiments.filter(e => e.status === 'completed').length },
    { key: 'running', label: 'Running', count: experiments.filter(e => e.status === 'running').length },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Experiments</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {filters.map(f => (
          <button key={f.key} onClick={() => setFilter(f.key)}
                  className={`px-4 py-3 rounded-lg border text-left transition-colors ${
                    filter === f.key
                      ? 'bg-blue-900/40 border-blue-700 text-blue-200'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  }`}>
            <p className="text-lg font-bold">{f.count}</p>
            <p className="text-xs capitalize">{f.label}</p>
          </button>
        ))}
      </div>

      <div className="bg-slate-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-900/50">
                <th className="text-left py-3 px-4">ID</th>
                <th className="text-left py-3 px-4">Type</th>
                <th className="text-left py-3 px-4">Status</th>
                <th className="text-right py-3 px-4">Score</th>
                <th className="text-right py-3 px-4">Steps</th>
                <th className="text-right py-3 px-4">Runtime</th>
                <th className="text-right py-3 px-4">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((exp) => (
                <tr key={exp.id} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                  <td className="py-3 px-4">
                    <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline font-mono text-xs">
                      {exp.id.length > 40 ? exp.id.slice(0, 40) + '...' : exp.id}
                    </a>
                  </td>
                  <td className="py-3 px-4"><SourceBadge source={exp.source} /></td>
                  <td className="py-3 px-4"><StatusBadge status={exp.status} /></td>
                  <td className="py-3 px-4 text-right font-mono">
                    {exp.final_score !== null ? (
                      <span className={exp.final_score > 0.5 ? 'text-emerald-400' : 'text-red-400'}>
                        {exp.final_score.toFixed(4)}
                      </span>
                    ) : <span className="text-slate-500">—</span>}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">{exp.total_steps ?? exp.steps?.length ?? 0}</td>
                  <td className="py-3 px-4 text-right text-slate-400">{exp.runtime ?? '—'}</td>
                  <td className="py-3 px-4 text-right text-slate-400 text-xs">
                    {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length === 0 && (
          <p className="text-slate-400 text-center py-8">No experiments match the current filter.</p>
        )}
      </div>
    </div>
  );
}
