'use client';

import { useState, useEffect, useMemo } from 'react';
import { SourceBadge, StatusBadge, getExpColor } from '../../src/components/StatCard';

interface Iteration {
  iteration: number;
  test_mae: number | null;
  status: string;
  hypothesis: string;
}

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  timestamp: string | null;
  source?: string;
  details?: { kept?: number; discarded?: number; best_mae?: number; first_mae?: number };
  iterations?: Iteration[];
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [search, setSearch] = useState('');

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

  const filtered = useMemo(() => {
    if (!search) return experiments;
    const q = search.toLowerCase();
    return experiments.filter(e => e.id.toLowerCase().includes(q));
  }, [experiments, search]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">🤒 Flu Experiments</h1>
        <input type="text" placeholder="Search..."
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
                <th className="text-right py-3 px-4">Best MAE</th>
                <th className="text-right py-3 px-4">Kept/Disc</th>
                <th className="text-right py-3 px-4">Iters</th>
                <th className="text-right py-3 px-4">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((exp, i) => {
                const d = exp.details || {};
                const bestMae = d.best_mae;
                const kept = d.kept ?? 0;
                const discarded = d.discarded ?? 0;
                const iterCount = exp.iterations?.length ?? 0;
                return (
                  <tr key={exp.id} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="py-3 px-4">
                      <a href={`/experiments/${exp.id}`}
                         className="text-blue-400 hover:underline font-mono text-xs">
                        {exp.id.length > 40 ? exp.id.slice(0, 40) + '...' : exp.id}
                      </a>
                    </td>
                    <td className="py-3 px-4"><SourceBadge source={exp.source} /></td>
                    <td className="py-3 px-4 text-right font-mono font-bold"
                        style={{color: bestMae != null ? '#22c55e' : '#64748b'}}>
                      {bestMae != null ? bestMae.toFixed(2) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">
                      <span className="text-emerald-400">{kept}</span>
                      <span className="text-slate-500"> / </span>
                      <span className="text-red-400">{discarded}</span>
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">{iterCount}</td>
                    <td className="py-3 px-4 text-right text-slate-400 text-xs">{exp.timestamp || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {filtered.length === 0 && (
          <p className="text-slate-400 text-center py-8">No flu experiments found.</p>
        )}
      </div>
    </div>
  );
}
