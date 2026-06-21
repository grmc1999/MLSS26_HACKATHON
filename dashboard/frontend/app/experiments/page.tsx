'use client';

import { useState, useEffect } from 'react';

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  total_steps: number;
  timestamp: string | null;
  runtime?: string;
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    async function fetchExperiments() {
      const res = await fetch('/api/experiments');
      const data = await res.json();
      setExperiments(data.experiments || []);
    }
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval);
  }, []);

  const filtered = experiments.filter((e) => filter === 'all' || e.status === filter);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Experiments</h1>

      <div className="flex gap-2">
        {['all', 'running', 'completed', 'pending'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded text-sm capitalize ${filter === f ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="bg-slate-800 rounded-lg p-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-2">ID</th>
              <th className="text-left py-2">Status</th>
              <th className="text-left py-2">Steps</th>
              <th className="text-left py-2">Score</th>
              <th className="text-left py-2">Runtime</th>
              <th className="text-left py-2">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((exp) => (
              <tr key={exp.id} className="border-b border-slate-800 hover:bg-slate-700">
                <td className="py-2">
                  <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline">
                    {exp.id}
                  </a>
                </td>
                <td className="py-2">
                  <span className={`px-2 py-1 rounded text-xs ${exp.status === 'completed' ? 'bg-green-900 text-green-200' : exp.status === 'running' ? 'bg-yellow-900 text-yellow-200' : 'bg-slate-700 text-slate-300'}`}>
                    {exp.status}
                  </span>
                </td>
                <td className="py-2">{exp.total_steps ?? 0}</td>
                <td className="py-2">{exp.final_score?.toFixed(4) ?? 'N/A'}</td>
                <td className="py-2">{exp.runtime ?? 'N/A'}</td>
                <td className="py-2 text-slate-400">{exp.timestamp ?? 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <p className="text-slate-400 text-center py-4">No experiments found.</p>
        )}
      </div>
    </div>
  );
}
