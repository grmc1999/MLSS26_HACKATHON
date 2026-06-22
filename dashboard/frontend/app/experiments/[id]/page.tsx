'use client';

import { useState, useEffect } from 'react';
import { use } from 'react';
import { StatCard, SourceBadge, StatusBadge, ScoreChart } from '../../../src/components/StatCard';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, Cell, ScatterChart, Scatter, ZAxis,
} from 'recharts';

interface Iteration {
  iteration: number;
  commit: string;
  test_acc: number | null;
  ood_f1: number | null;
  memory_gb: string;
  status: string;
  description: string;
}

interface Embedding {
  x: number;
  y: number;
  label: number;
  label_name: string;
  pred_raw: number;
  pred_ood: number;
  is_ood: boolean;
}

interface VizData {
  per_class_accuracy?: Record<string, { total: number; correct: number; accuracy: number }>;
  confusion_matrix?: number[][];
  ood_confusion_matrix?: number[][];
  class_names?: string[];
  embeddings?: Embedding[];
  sample_images?: Record<string, string[]>;
  pca_explained_variance?: number[];
  total_samples?: number;
  test_acc?: number;
  ood_f1?: number;
}

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
  iterations?: Iteration[];
  runtime?: string;
  agent_log_file?: string | null;
}

const STATUS_COLORS: Record<string, string> = {
  keep: '#10b981',
  discard: '#ef4444',
  baseline: '#3b82f6',
};

export default function ExperimentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [expRes, vizRes] = await Promise.all([
          fetch(`/api/experiments/${encodeURIComponent(id)}`),
          fetch(`/api/experiments/${encodeURIComponent(id)}/viz`),
        ]);
        if (expRes.ok) {
          setExperiment(await expRes.json());
        } else {
          setExperiment(null);
        }
        if (vizRes.ok) {
          const vd = await vizRes.json();
          if (Object.keys(vd).length > 0) setVizData(vd);
        }
      } catch (e) {
        console.error('Failed to fetch experiment:', e);
        setExperiment(null);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 10000);
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
  const iterations = experiment.iterations || [];
  const improvement = scores.length >= 2 ? ((scores[scores.length - 1].score - scores[0].score)) : null;

  const keptIterations = iterations.filter(i => i.status === 'keep');
  const discardedIterations = iterations.filter(i => i.status === 'discard');

  const runDetails = experiment.source === 'run_exp' ? (experiment.details || {}) : {};

  const iterChartData = iterations.map((it, idx) => ({
    iteration: it.iteration,
    test_acc: it.test_acc,
    ood_f1: it.ood_f1,
    status: it.status,
    delta: idx > 0 && iterations[idx - 1].test_acc
      ? it.test_acc !== null && iterations[idx - 1].test_acc !== null
        ? (it.test_acc - iterations[idx - 1].test_acc) : 0
      : 0,
  }));

  const bestTestAcc = Math.max(...iterChartData.map(d => d.test_acc ?? 0));
  const bestOODF1 = Math.max(...iterChartData.map(d => d.ood_f1 ?? 0));
  const valAcc = runDetails.val_acc as number | undefined;
  const idTestAcc = runDetails.test_acc_id as number | undefined;

  const accuracyBarData = iterations
    .filter(i => i.test_acc !== null && i.ood_f1 !== null)
    .map(i => ({
      name: `#${i.iteration}`,
      test_acc: i.test_acc,
      ood_f1: i.ood_f1,
      status: i.status,
    }));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 flex-wrap">
        <a href="/experiments" className="text-blue-400 hover:underline text-sm">← Experiments</a>
        <h1 className="text-2xl font-bold font-mono text-sm truncate max-w-[500px]">{experiment.id}</h1>
        <SourceBadge source={experiment.source} />
        <StatusBadge status={experiment.status} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        <StatCard label="Val Acc (PneumoniaMNIST)" value={valAcc !== undefined ? valAcc.toFixed(4) : '—'}
                  color="#06b6d4" />
        <StatCard label="Test ID Acc (Normal+Pneu)" value={idTestAcc !== undefined ? idTestAcc.toFixed(4) : '—'}
                  color="#10b981" />
        <StatCard label="Best Test 3-class Acc" value={bestTestAcc.toFixed(4)}
                  color="#3b82f6" />
        <StatCard label="Best OOD F1" value={bestOODF1.toFixed(4)}
                  color="#8b5cf6" />
        <StatCard label="Total Iterations" value={iterations.length} />
        <StatCard label="Kept / Discarded"
                  value={`${keptIterations.length} / ${discardedIterations.length}`}
                  color={keptIterations.length > discardedIterations.length ? '#10b981' : '#f59e0b'} />
        <StatCard label="Status" value={experiment.status} />
      </div>

      {iterChartData.length > 1 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Test Accuracy & OOD F1 Over Iterations</h2>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={iterChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="iteration" stroke="#94a3b8" label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={[0, 1]} />
              <YAxis yAxisId="right" orientation="right" stroke="#8b5cf6" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }} />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="test_acc" stroke="#3b82f6" strokeWidth={2.5}
                    dot={{ r: 5 }} name="Test Accuracy" connectNulls />
              <Line yAxisId="right" type="monotone" dataKey="ood_f1" stroke="#8b5cf6" strokeWidth={2.5}
                    dot={{ r: 5 }} name="OOD F1" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {accuracyBarData.length > 1 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Per-Iteration Metrics Comparison</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyBarData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }} />
              <Legend />
              <Bar dataKey="test_acc" fill="#3b82f6" name="Test Accuracy" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ood_f1" fill="#8b5cf6" name="OOD F1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {iterations.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-3">Iteration Log</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 bg-slate-900/50">
                  <th className="text-left py-2 px-3">#</th>
                  <th className="text-left py-2 px-3">Commit</th>
                  <th className="text-right py-2 px-3">Test Acc</th>
                  <th className="text-right py-2 px-3">OOD F1</th>
                  <th className="text-right py-2 px-3">Memory</th>
                  <th className="text-left py-2 px-3">Status</th>
                  <th className="text-left py-2 px-3">Description</th>
                </tr>
              </thead>
              <tbody>
                {iterations.map((it, idx) => {
                  const prevTestAcc = idx > 0 ? iterations[idx - 1].test_acc : null;
                  const testDelta = it.test_acc !== null && prevTestAcc !== null
                    ? it.test_acc - prevTestAcc : null;
                  return (
                    <tr key={idx} className="border-b border-slate-800 hover:bg-slate-700/30">
                      <td className="py-2 px-3 font-mono text-slate-400">{it.iteration}</td>
                      <td className="py-2 px-3 font-mono text-xs text-slate-400">
                        {it.commit.length > 12 ? it.commit.slice(0, 12) + '...' : it.commit}
                      </td>
                      <td className={`py-2 px-3 text-right font-mono ${testDelta !== null && testDelta > 0 ? 'text-emerald-400' : testDelta !== null && testDelta < 0 ? 'text-red-400' : 'text-white'}`}>
                        {it.test_acc !== null ? it.test_acc.toFixed(4) : '—'}
                        {testDelta !== null && (
                          <span className={`text-xs ml-1 ${testDelta >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                            ({testDelta >= 0 ? '+' : ''}{testDelta.toFixed(4)})
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-white">
                        {it.ood_f1 !== null ? it.ood_f1.toFixed(4) : '—'}
                      </td>
                      <td className="py-2 px-3 text-right text-slate-400">{it.memory_gb}</td>
                      <td className="py-2 px-3">
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          it.status === 'keep' ? 'bg-emerald-900/50 text-emerald-300' :
                          it.status === 'discard' ? 'bg-red-900/50 text-red-300' :
                          'bg-blue-900/50 text-blue-300'
                        }`}>
                          {it.status}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-300 text-xs max-w-[300px] truncate">{it.description}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {vizData && vizData.sample_images && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Sample Images by Class</h2>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(vizData.sample_images).map(([className, images]) => (
              <div key={className}>
                <h3 className="text-sm font-medium text-slate-300 mb-2 capitalize">{className}</h3>
                <div className="grid grid-cols-3 gap-2">
                  {(images as string[]).map((b64, i) => (
                    <img key={i} src={`data:image/png;base64,${b64}`}
                         alt={`${className} sample ${i}`}
                         className="rounded border border-slate-600 w-full aspect-square object-cover" />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {vizData && vizData.embeddings && vizData.embeddings.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-2">Feature Embeddings (PCA)</h2>
          <p className="text-xs text-slate-400 mb-4">
            Penultimate layer features projected to 2D via PCA.
            Each point is a test sample colored by ground truth class.
          </p>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="x" stroke="#94a3b8" name="PC1" />
              <YAxis dataKey="y" stroke="#94a3b8" name="PC2" />
              <ZAxis range={[40, 60]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                       formatter={(value: number, name: string) => [value.toFixed(2), name]}
                       labelFormatter={(label: string) => ''} />
              <Legend />
              {[0, 1, 2].map(cls => {
                const color = cls === 0 ? '#3b82f6' : cls === 1 ? '#10b981' : '#ef4444';
                const name = vizData.class_names?.[cls] ?? `Class ${cls}`;
                const data = vizData.embeddings?.filter(e => e.label === cls) ?? [];
                return <Scatter key={cls} name={name} data={data} fill={color} shape="circle" />;
              })}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {vizData && vizData.per_class_accuracy && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Per-Class Accuracy</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={(Object.entries(vizData.per_class_accuracy) as [string, { accuracy: number }][]).map(([name, d]) => ({
              name,
              accuracy: d.accuracy,
            }))} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }} />
              <Bar dataKey="accuracy" name="Accuracy" radius={[4, 4, 0, 0]}>
                {(Object.entries(vizData.per_class_accuracy || {}) as [string, { accuracy: number }][]).map(([name, d], i) => (
                  <Cell key={name} fill={d.accuracy > 0.5 ? '#10b981' : d.accuracy > 0.2 ? '#f59e0b' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {vizData && vizData.ood_confusion_matrix && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">OOD Confusion Matrix</h2>
          <p className="text-xs text-slate-400 mb-3">
            Rows = true class, Columns = predicted class (0: normal, 1: pneumonia, 2: OOD/consolidation)
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="py-2 px-3 text-left">True ↓ / Pred →</th>
                  <th className="py-2 px-3 text-right">Normal (0)</th>
                  <th className="py-2 px-3 text-right">Pneumonia (1)</th>
                  <th className="py-2 px-3 text-right">OOD (2)</th>
                </tr>
              </thead>
              <tbody>
                {vizData.ood_confusion_matrix.map((row, ri) => (
                  <tr key={ri} className="border-b border-slate-800">
                    <td className="py-2 px-3 font-medium text-slate-300">
                      {vizData.class_names?.[ri] ?? ri} ({ri})
                    </td>
                    {row.map((val, ci) => (
                      <td key={ci} className={`py-2 px-3 text-right font-mono ${
                        ri === ci ? 'text-emerald-400' : 'text-red-400'
                      }`}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Score Progression</h2>
        <ScoreChart scores={scores} color="#3b82f6" height={300} />
      </div>

      {experiment.source === 'run_exp' && experiment.details && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-3">Run Details</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {Object.entries(experiment.details).filter(([k]) => k !== 'iterations').map(([key, value]) => (
              <div key={key}>
                <span className="text-slate-400 block">{key.replace('_', ' ')}</span>
                <span className="text-white font-mono">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

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
    </div>
  );
}
