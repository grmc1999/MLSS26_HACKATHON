'use client';

import { useState, useEffect } from 'react';
import { use } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ExperimentDetail {
  id: string;
  status: string;
  final_score: number | null;
  total_steps: number;
  scores: { step: number; score: number }[];
  steps: { step: number; action: any; observation: string }[];
  runtime?: string;
  timestamp: string | null;
}

export default function ExperimentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);

  useEffect(() => {
    async function fetchExperiment() {
      const res = await fetch(`/api/experiments/${id}`);
      if (res.ok) {
        setExperiment(await res.json());
      }
    }
    fetchExperiment();
    const interval = setInterval(fetchExperiment, 5000);
    return () => clearInterval(interval);
  }, [id]);

  if (!experiment) {
    return <div className="text-slate-400">Loading experiment {id}...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Experiment: {experiment.id}</h1>
        <span className={`px-3 py-1 rounded text-sm ${experiment.status === 'completed' ? 'bg-green-900 text-green-200' : experiment.status === 'running' ? 'bg-yellow-900 text-yellow-200' : 'bg-slate-700 text-slate-300'}`}>
          {experiment.status}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard label="Total Steps" value={experiment.total_steps ?? 0} />
        <StatCard label="Final Score" value={experiment.final_score?.toFixed(4) ?? 'N/A'} />
        <StatCard label="Runtime" value={experiment.runtime ?? 'N/A'} />
        <StatCard label="Timestamp" value={experiment.timestamp ?? 'N/A'} />
      </div>

      {experiment.scores.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Score Progression</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={experiment.scores}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
              <Line type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Agent Activity Log</h2>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {experiment.steps.map((step, i) => (
            <div key={i} className="border-l-2 border-blue-500 pl-4 py-2">
              <div className="text-sm font-semibold text-blue-400">Step {step.step}</div>
              <div className="text-xs text-slate-400 mt-1">
                Action: {JSON.stringify(step.action)}
              </div>
              <div className="text-xs text-slate-500 mt-1 font-mono">
                {step.observation}
              </div>
            </div>
          ))}
          {experiment.steps.length === 0 && (
            <p className="text-slate-400">No steps recorded yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <div className="text-slate-400 text-sm">{label}</div>
      <div className="text-2xl font-bold mt-2">{value}</div>
    </div>
  );
}
