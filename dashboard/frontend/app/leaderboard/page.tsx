'use client';

import { useState, useEffect, useMemo } from 'react';
import { SourceBadge, StatusBadge } from '../../src/components/StatCard';

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  total_steps?: number;
  timestamp: string | null;
  source?: string;
  runtime?: string;
}

const MEDALS = ['🥇', '🥈', '🥉'];

export default function LeaderboardPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        setExperiments(data.experiments || []);
      } catch (e) {
        console.error(e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const ranked = useMemo(() =>
    experiments
      .filter(e => e.final_score !== null)
      .sort((a, b) => (b.final_score as number) - (a.final_score as number)),
    [experiments]
  );

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Leaderboard</h1>

      <div className="bg-slate-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-900/50">
                <th className="text-left py-3 px-4 w-12">Rank</th>
                <th className="text-left py-3 px-4">Experiment</th>
                <th className="text-left py-3 px-4">Type</th>
                <th className="text-right py-3 px-4">Score</th>
                <th className="text-right py-3 px-4">Steps</th>
                <th className="text-right py-3 px-4">Runtime</th>
                <th className="text-right py-3 px-4">Date</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((exp, i) => (
                <tr key={exp.id}
                    className={`border-b border-slate-700/50 hover:bg-slate-700/30 ${
                      i === 0 ? 'bg-yellow-900/10' : ''
                    }`}>
                  <td className="py-3 px-4 text-lg">
                    {i < 3 ? MEDALS[i] : <span className="text-slate-500 text-sm">{i + 1}</span>}
                  </td>
                  <td className="py-3 px-4">
                    <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline font-mono text-xs">
                      {exp.id.length > 40 ? exp.id.slice(0, 40) + '...' : exp.id}
                    </a>
                  </td>
                  <td className="py-3 px-4"><SourceBadge source={exp.source} /></td>
                  <td className={`py-3 px-4 text-right font-mono text-lg font-bold ${
                    exp.final_score && exp.final_score > 0.5 ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {exp.final_score?.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">{exp.total_steps ?? 0}</td>
                  <td className="py-3 px-4 text-right text-slate-400">{exp.runtime ?? '—'}</td>
                  <td className="py-3 px-4 text-right text-slate-400 text-xs">
                    {exp.timestamp ? new Date(exp.timestamp).toLocaleDateString() : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {ranked.length === 0 && (
          <p className="text-slate-400 text-center py-8">No scored experiments yet.</p>
        )}
      </div>
    </div>
  );
}
