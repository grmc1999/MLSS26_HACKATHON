'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface Experiment {
  id: string;
  status: string;
  final_score: number | null;
  total_steps: number;
  timestamp: string | null;
}

interface ScoreData {
  step: number;
  score: number;
}

interface StatusData {
  total_experiments: number;
  total_agents: number;
  logs_exist: boolean;
  timestamp: string;
}

export default function Home() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [allScores, setAllScores] = useState<Record<string, ScoreData[]>>({});

  useEffect(() => {
    async function fetchData() {
      try {
        const statusRes = await fetch('/api/status');
        setStatus(await statusRes.json());

        const expRes = await fetch('/api/experiments');
        const expData = await expRes.json();
        setExperiments(expData.experiments || []);

        const scoresRes = await fetch('/api/scores');
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

  const chartData = Object.entries(allScores).flatMap(([expId, scores]) =>
    scores.map((s) => ({ experiment: expId, step: s.step, score: s.score }))
  );

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Dashboard Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard label="Experiments" value={status?.total_experiments ?? 0} />
        <StatCard label="Agents" value={status?.total_agents ?? 0} />
        <StatCard
          label="Best Score"
          value={experiments
            .map((e) => e.final_score)
            .filter((s): s is number => s !== null)
            .sort((a, b) => b - a)[0]
            ?.toFixed(4) ?? 'N/A'}
        />
        <StatCard label="Running" value={experiments.filter((e) => e.status === 'running').length} />
      </div>

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Score Timeline</h2>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" label={{ value: 'Step', position: 'insideBottom' }} />
              <YAxis stroke="#94a3b8" label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
              <Legend />
              {Object.keys(allScores).map((expId) => (
                <Line
                  key={expId}
                  type="monotone"
                  dataKey="score"
                  data={chartData.filter((d) => d.experiment === expId)}
                  name={expId}
                  stroke={`#${Math.floor(Math.random() * 16777215).toString(16)}`}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-slate-400">No score data available yet. Run an experiment to see results.</p>
        )}
      </div>

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Experiments</h2>
        {experiments.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-2">ID</th>
                <th className="text-left py-2">Status</th>
                <th className="text-left py-2">Steps</th>
                <th className="text-left py-2">Score</th>
                <th className="text-left py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {experiments.slice(0, 10).map((exp) => (
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
                  <td className="py-2 text-slate-400">{exp.timestamp ?? 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-slate-400">No experiments found. Run the hackathon script to start.</p>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <div className="text-slate-400 text-sm">{label}</div>
      <div className="text-3xl font-bold mt-2">{value}</div>
    </div>
  );
}
