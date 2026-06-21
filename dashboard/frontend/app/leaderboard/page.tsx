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

export default function LeaderboardPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    async function fetchExperiments() {
      const res = await fetch('/api/experiments');
      const data = await res.json();
      const exps = (data.experiments || []).filter((e: Experiment) => e.final_score !== null);
      exps.sort((a: Experiment, b: Experiment) => (b.final_score ?? 0) - (a.final_score ?? 0));
      setExperiments(exps);
    }
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Leaderboard</h1>
      <p className="text-slate-400">Ranked by final score (Dice Score for contrail detection)</p>

      <div className="bg-slate-800 rounded-lg p-6">
        {experiments.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-2">Rank</th>
                <th className="text-left py-2">Experiment ID</th>
                <th className="text-left py-2">Score</th>
                <th className="text-left py-2">Steps</th>
                <th className="text-left py-2">Runtime</th>
                <th className="text-left py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, idx) => (
                <tr key={exp.id} className={`border-b border-slate-800 ${idx === 0 ? 'bg-yellow-900/20' : ''}`}>
                  <td className="py-2">
                    {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : idx + 1}
                  </td>
                  <td className="py-2">
                    <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline">
                      {exp.id}
                    </a>
                  </td>
                  <td className="py-2 font-bold text-green-400">
                    {exp.final_score?.toFixed(6)}
                  </td>
                  <td className="py-2">{exp.total_steps}</td>
                  <td className="py-2">{exp.runtime ?? 'N/A'}</td>
                  <td className="py-2 text-slate-400">{exp.timestamp ?? 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-slate-400 text-center py-4">No completed experiments with scores yet.</p>
        )}
      </div>
    </div>
  );
}
