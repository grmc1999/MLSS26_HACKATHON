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
  iterations?: { iteration: number; test_acc: number | null; ood_f1: number | null; status: string }[];
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [search, setSearch] = useState('');

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
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const filtered = useMemo(() => {
    let filtered = experiments;
    if (filter === 'loops') filtered = filtered.filter(e => e.source === 'auto_loop');
    else if (filter === 'runs') filtered = filtered.filter(e => e.source === 'run_exp');
    else if (filter === 'completed') filtered = filtered.filter(e => e.status === 'completed');
    else if (filter === 'running') filtered = filtered.filter(e => e.status === 'running');
    if (search) {
      const q = search.toLowerCase();
      filtered = filtered.filter(e => e.id.toLowerCase().includes(q));
    }
    return filtered;
  }, [experiments, filter, search]);

  const stats = useMemo(() => ({
    total: experiments.length,
    loops: experiments.filter(e => e.source === 'auto_loop').length,
    runs: experiments.filter(e => e.source === 'run_exp').length,
    logs: experiments.filter(e => !e.source || e.source === 'logs').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    running: experiments.filter(e => e.status === 'running').length,
  }), [experiments]);

  const filters = [
    { key: 'all', label: 'All', count: stats.total },
    { key: 'loops', label: 'Auto Loops', count: stats.loops },
    { key: 'runs', label: 'Run Experiments', count: stats.runs },
    { key: 'completed', label: 'Completed', count: stats.completed },
    { key: 'running', label: 'Running', count: stats.running },
  ];

  const bestTestAcc = useMemo(() => {
    let best = 0;
    experiments.forEach(exp => {
      if (exp.iterations) {
        exp.iterations.forEach(it => {
          if (it.test_acc && it.test_acc > best) best = it.test_acc;
        });
      }
      if (exp.final_score && exp.final_score > best) best = exp.final_score;
    });
    return best;
  }, [experiments]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Experiments</h1>
        <div className="text-sm text-slate-400">
          Best Test Acc: <span className="text-emerald-400 font-mono font-bold">{bestTestAcc.toFixed(4)}</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center">
        {filters.map(f => (
          <button key={f.key} onClick={() => setFilter(f.key)}
                  className={`px-4 py-2.5 rounded-lg border text-sm transition-colors ${
                    filter === f.key
                      ? 'bg-blue-900/40 border-blue-700 text-blue-200'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  }`}>
            <span className="font-bold mr-1">{f.count}</span>
            <span>{f.label}</span>
          </button>
        ))}
        <div className="flex-1" />
        <input type="text" placeholder="Search experiments..."
               value={search} onChange={e => setSearch(e.target.value)}
               className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 w-64" />
      </div>

      <div className="bg-slate-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-900/50">
                <th className="text-left py-3 px-4">ID</th>
                <th className="text-left py-3 px-4">Type</th>
                <th className="text-left py-3 px-4">Status</th>
                <th className="text-right py-3 px-4">Best Test Acc</th>
                <th className="text-right py-3 px-4">Best OOD F1</th>
                <th className="text-right py-3 px-4">Iterations</th>
                <th className="text-right py-3 px-4">Steps</th>
                <th className="text-right py-3 px-4">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((exp) => {
                const bestAcc = exp.iterations
                  ? Math.max(...exp.iterations.map(i => i.test_acc ?? 0))
                  : (exp.final_score ?? 0);
                const bestF1 = exp.iterations
                  ? Math.max(...exp.iterations.map(i => i.ood_f1 ?? 0))
                  : 0;
                const iterCount = exp.iterations?.length ?? 0;
                const keptCount = exp.iterations?.filter(i => i.status === 'keep').length ?? 0;
                return (
                  <tr key={exp.id} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-3 px-4">
                      <a href={`/experiments/${exp.id}`}
                         className="text-blue-400 hover:underline font-mono text-xs">
                        {exp.id.length > 40 ? exp.id.slice(0, 40) + '...' : exp.id}
                      </a>
                    </td>
                    <td className="py-3 px-4"><SourceBadge source={exp.source} /></td>
                    <td className="py-3 px-4"><StatusBadge status={exp.status} /></td>
                    <td className="py-3 px-4 text-right font-mono">
                      <span className={bestAcc > 0.25 ? 'text-emerald-400' : 'text-red-400'}>
                        {bestAcc.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono">
                      <span className={bestF1 > 0.1 ? 'text-purple-400' : 'text-red-400'}>
                        {bestF1 > 0 ? bestF1.toFixed(4) : '—'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">
                      {iterCount > 0 ? `${iterCount} (${keptCount} kept)` : '—'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">{exp.total_steps ?? exp.steps?.length ?? 0}</td>
                    <td className="py-3 px-4 text-right text-slate-400 text-xs">
                      {exp.timestamp ? new Date(exp.timestamp).toLocaleString() : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {filtered.length === 0 && (
          <p className="text-slate-400 text-center py-8">
            {search ? 'No experiments match your search.' : 'No experiments match the current filter.'}
          </p>
        )}
      </div>
    </div>
  );
}
