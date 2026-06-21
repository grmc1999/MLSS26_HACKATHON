'use client';

import { useState, useEffect } from 'react';
import { use } from 'react';
import { StatCard, SourceBadge, StatusBadge, ScoreChart } from '../../../src/components/StatCard';

interface ExperimentDetail {
  id: string;
  status: string;
  final_score: number | null;
  total_steps?: number;
  timestamp: string | null;
  source?: string;
  scores?: { step: number; score: number }[];
  steps?: { step: number; action?: Record<string, unknown>; observation?: string }[];
  details?: Record<string, unknown>;
  runtime?: string;
  agent_log_file?: string | null;
}

export default function ExperimentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`/api/experiments/${encodeURIComponent(id)}`);
        if (res.ok) {
          setExperiment(await res.json());
        } else {
          setExperiment(null);
        }
      } catch (e) {
        console.error('Failed to fetch experiment:', e);
        setExperiment(null);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <p className="text-slate-400">Loading experiment...</p>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="text-center py-20">
        <p className="text-slate-400 text-lg">Experiment not found</p>
        <a href="/experiments" className="text-blue-400 hover:underline mt-4 inline-block">← Back to experiments</a>
      </div>
    );
  }

  const scores = experiment.scores || [];
  const steps = experiment.steps || [];
  const improvement = scores.length >= 2 ? ((scores[scores.length - 1].score - scores[0].score)) : null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 flex-wrap">
        <a href="/experiments" className="text-blue-400 hover:underline text-sm">← Experiments</a>
        <h1 className="text-2xl font-bold font-mono text-sm truncate max-w-[500px]">{experiment.id}</h1>
        <SourceBadge source={experiment.source} />
        <StatusBadge status={experiment.status} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <StatCard label="Final Score" value={experiment.final_score !== null ? experiment.final_score.toFixed(4) : 'N/A'}
                  color={experiment.final_score !== null && experiment.final_score > 0.5 ? '#10b981' : '#ef4444'} />
        <StatCard label="Steps" value={experiment.total_steps ?? steps.length ?? 0} />
        <StatCard label="Best Improvement" value={improvement !== null ? `${(improvement * 100).toFixed(1)}%` : 'N/A'} />
        <StatCard label="Runtime" value={experiment.runtime || '—'} />
        <StatCard label="Status" value={experiment.status} />
        <StatCard label="Timestamp" value={experiment.timestamp ? new Date(experiment.timestamp).toLocaleDateString() : '—'} />
      </div>

      {experiment.source === 'run_exp' && experiment.details && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-3">Run Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {Object.entries(experiment.details).map(([key, value]) => (
              <div key={key}>
                <span className="text-slate-400 block">{key.replace('_', ' ')}</span>
                <span className="text-white font-mono">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Score Progression</h2>
        <ScoreChart scores={scores} color="#3b82f6" height={300} />
      </div>

      {steps.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Agent Activity Log ({steps.length} steps)</h2>
          <div className="max-h-96 overflow-y-auto space-y-3">
            {steps.map((step, i) => (
              <div key={i} className="border-l-2 border-blue-600 pl-4 py-2">
                <p className="text-xs text-slate-500 mb-1">Step {step.step ?? i}</p>
                {step.action && Object.keys(step.action).length > 0 && (
                  <pre className="text-xs text-slate-300 bg-slate-900 rounded p-2 mb-1 overflow-x-auto">
                    {JSON.stringify(step.action, null, 1).slice(0, 300)}
                  </pre>
                )}
                {step.observation && (
                  <p className="text-xs text-slate-400 line-clamp-3">{step.observation.slice(0, 500)}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {experiment.source === 'auto_loop' && scores.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-3">Iteration Log</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-2 px-2">#</th>
                  <th className="text-right py-2 px-2">Score</th>
                  <th className="text-right py-2 px-2">Delta</th>
                </tr>
              </thead>
              <tbody>
                {scores.map((s, i) => (
                  <tr key={i} className="border-b border-slate-800">
                    <td className="py-1.5 px-2">{s.step}</td>
                    <td className={`py-1.5 px-2 text-right font-mono ${s.score > 0.5 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {s.score.toFixed(4)}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-slate-400">
                      {i > 0 ? (s.score - scores[i - 1].score >= 0 ? '+' : '') + (s.score - scores[i - 1].score).toFixed(4) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
