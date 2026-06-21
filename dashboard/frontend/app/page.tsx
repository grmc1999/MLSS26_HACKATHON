'use client';

import { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { StatCard, SourceBadge, StatusBadge, getExpColor, ScoreChart } from '../src/components/StatCard';

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  total_steps?: number;
  timestamp: string | null;
  source?: string;
  scores?: { step: number; score: number }[];
  steps?: { step: number }[];
  details?: Record<string, unknown>;
}

interface ScoreData { step: number; score: number; }
interface StatusData {
  total_experiments: number;
  total_agents: number;
  logs_exist: boolean;
  timestamp: string;
}

const EXPERIMENT_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

export default function Home() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [allScores, setAllScores] = useState<Record<string, ScoreData[]>>({});

  useEffect(() => {
    async function fetchData() {
      try {
        const [statusRes, expRes, scoresRes] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/experiments'),
          fetch('/api/scores'),
        ]);
        setStatus(await statusRes.json());
        const expData = await expRes.json();
        setExperiments(expData.experiments || []);
        const scoresData = await scoresRes.json();
        setAllScores(scoresData.all_scores || {});
      } catch (e) {
        console.error('Failed to fetch data:', e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const bestScore = useMemo(() =>
    experiments.map(e => e.final_score).filter((s): s is number => s !== null).sort((a, b) => b - a)[0] ?? null,
    [experiments]
  );

  const runningCount = useMemo(() =>
    experiments.filter(e => e.status === 'running').length,
    [experiments]
  );

  const scoreKeys = useMemo(() => Object.keys(allScores), [allScores]);

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Dashboard Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <StatCard label="Total Experiments" value={status?.total_experiments ?? 0}
                  sub={`${experiments.filter(e => e.source === 'auto_loop').length} auto loops`} />
        <StatCard label="Agents" value={status?.total_agents ?? 0} />
        <StatCard label="Best Score" value={bestScore !== null ? bestScore.toFixed(4) : 'N/A'}
                  color="#10b981" sub={bestScore !== null ? 'Test Dice' : undefined} />
        <StatCard label="Running" value={runningCount} color="#f59e0b" />
        <StatCard label="Improvement" value={
          (() => {
            const scores = experiments.map(e => e.final_score).filter((s): s is number => s !== null);
            if (scores.length < 2) return 'N/A';
            return `${(((scores[0] - scores[scores.length - 1]) / Math.max(scores[scores.length - 1], 0.0001)) * 100).toFixed(0)}%`;
          })()
        } color="#8b5cf6" sub="first → latest" />
      </div>

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Score Timeline</h2>
        {scoreKeys.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" type="number" domain={['dataMin', 'dataMax']} />
              <YAxis stroke="#94a3b8" domain={[0, 'auto']} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }} />
              <Legend />
              {scoreKeys.map((expId, i) => (
                <Line key={expId} type="monotone" dataKey="score"
                      data={allScores[expId]} name={expId.slice(0, 28)}
                      stroke={getExpColor(expId, i)} dot={false} strokeWidth={2}
                      connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-slate-400 text-center py-8">No score data available yet.</p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Experiments</h2>
          {experiments.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2">ID</th>
                    <th className="text-left py-2">Type</th>
                    <th className="text-left py-2">Status</th>
                    <th className="text-right py-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.slice(0, 12).map((exp) => (
                    <tr key={exp.id} className="border-b border-slate-800 hover:bg-slate-700/50">
                      <td className="py-2 max-w-[180px] truncate">
                        <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline font-mono text-xs">
                          {exp.id}
                        </a>
                      </td>
                      <td className="py-2"><SourceBadge source={exp.source} /></td>
                      <td className="py-2"><StatusBadge status={exp.status} /></td>
                      <td className="py-2 text-right font-mono">
                        {exp.final_score !== null ? exp.final_score.toFixed(4) : <span className="text-slate-500">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-slate-400">No experiments found.</p>
          )}
        </div>

        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Auto-Loop Experiments</h2>
          {(() => {
            const loops = experiments.filter(e => e.source === 'auto_loop');
            if (loops.length === 0) return <p className="text-slate-400">No auto-loop experiments yet. Run the autoresearch loop.</p>;
            return (
              <div className="space-y-4">
                {loops.slice(0, 3).map((loop) => (
                  <div key={loop.id} className="bg-slate-900 rounded p-4 border border-slate-700">
                    <div className="flex justify-between items-start mb-2">
                      <a href={`/experiments/${loop.id}`} className="text-purple-400 hover:underline font-mono text-sm">
                        {loop.id}
                      </a>
                      <span className="text-sm font-bold" style={{ color: loop.final_score !== null && loop.final_score > 0.5 ? '#10b981' : '#ef4444' }}>
                        {loop.final_score !== null ? loop.final_score.toFixed(4) : '—'}
                      </span>
                    </div>
                    <ScoreChart scores={loop.scores || []} color="#8b5cf6" height={120} />
                  </div>
                ))}
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
