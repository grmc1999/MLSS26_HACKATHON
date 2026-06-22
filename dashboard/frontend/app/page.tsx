'use client';

import { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar,
} from 'recharts';
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
  iterations?: { iteration: number; val_acc?: number | null; test_acc_id?: number | null; test_acc: number | null; ood_f1: number | null; status: string }[];
}

interface ScoreData { step: number; score: number; }
interface StatusData {
  total_experiments: number;
  total_agents: number;
  logs_exist: boolean;
  timestamp: string;
}

export default function Home() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statusRes, expRes] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/experiments'),
        ]);
        setStatus(await statusRes.json());
        const expData = await expRes.json();
        setExperiments(expData.experiments || []);
      } catch (e) {
        console.error('Failed to fetch data:', e);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const loopExps = useMemo(() =>
    experiments.filter(e => e.source === 'auto_loop'),
    [experiments]
  );

  const recentExps = useMemo(() =>
    experiments.filter(e => e.source === 'run_exp').slice(0, 12),
    [experiments]
  );

  const allIterations = useMemo(() => {
    const data: { iteration: number; val_acc: number; test_acc_id: number; ood_f1: number; loop: string }[] = [];
    loopExps.forEach(exp => {
      if (exp.iterations) {
        exp.iterations.forEach(it => {
          if (it.test_acc_id !== null && it.ood_f1 !== null) {
            data.push({
              iteration: it.iteration,
              val_acc: it.val_acc ?? 0,
              test_acc_id: it.test_acc_id ?? 0,
              ood_f1: it.ood_f1 ?? 0,
              loop: exp.id,
            });
          }
        });
      }
    });
    return data.sort((a, b) => a.iteration - b.iteration);
  }, [loopExps]);

  const bestValAcc = useMemo(() =>
    Math.max(...allIterations.map(d => d.val_acc), 0),
    [allIterations]
  );

  const bestTestAcc = useMemo(() =>
    Math.max(...allIterations.map(d => d.test_acc_id), 0),
    [allIterations]
  );

  const bestOODF1 = useMemo(() =>
    Math.max(...allIterations.map(d => d.ood_f1), 0),
    [allIterations]
  );

  const latestTestAcc = allIterations.length > 0 ? allIterations[allIterations.length - 1].test_acc_id : 0;
  const latestOODF1 = allIterations.length > 0 ? allIterations[allIterations.length - 1].ood_f1 : 0;
  const firstTestAcc = allIterations.length > 0 ? allIterations[0].test_acc_id : 0;

  const recentRuns = useMemo(() =>
    experiments.filter(e => e.source === 'run_exp').slice(0, 8),
    [experiments]
  );

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Dashboard Overview</h1>

      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <StatCard label="Total Experiments" value={status?.total_experiments ?? 0}
                  sub={`${loopExps.length} auto loops`} />
        <StatCard label="Agents" value={status?.total_agents ?? 0} />
        <StatCard label="Best ID Test Acc" value={bestTestAcc.toFixed(4)}
                  color="#10b981" sub="ChestMNIST Normal+Pneumonia" />
        <StatCard label="Best OOD F1" value={bestOODF1.toFixed(4)}
                  color="#8b5cf6" sub="Consolidation detection" />
        <StatCard label="Best Val Acc" value={bestValAcc > 0 ? bestValAcc.toFixed(4) : 'N/A'}
                  color="#06b6d4" sub="PneumoniaMNIST validation" />
        <StatCard label="Improvement" value={
          firstTestAcc > 0 ? `${((latestTestAcc - firstTestAcc) / firstTestAcc * 100).toFixed(0)}%` : 'N/A'
        } color="#f59e0b" sub="ID test acc: first → latest" />
      </div>

      {allIterations.length > 1 && (
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Metrics Over All Iterations</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={allIterations} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="iteration" stroke="#94a3b8" label={{ value: 'Iteration', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }} />
              <Legend />
              <Line type="monotone" dataKey="val_acc" stroke="#06b6d4" strokeWidth={2}
                    dot={{ r: 5 }} name="Val Acc" connectNulls />
              <Line type="monotone" dataKey="test_acc_id" stroke="#10b981" strokeWidth={2.5}
                    dot={{ r: 5 }} name="ID Test Acc" connectNulls />
              <Line type="monotone" dataKey="ood_f1" stroke="#8b5cf6" strokeWidth={2.5}
                    dot={{ r: 5 }} name="OOD F1" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Runs</h2>
          {recentRuns.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2">ID</th>
                    <th className="text-right py-2">Val Acc</th>
                    <th className="text-right py-2">ID Test Acc</th>
                    <th className="text-right py-2">OOD F1</th>
                    <th className="text-right py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map((exp) => {
                    const details = exp.details || {};
                    return (
                      <tr key={exp.id} className="border-b border-slate-800 hover:bg-slate-700/50">
                        <td className="py-2 max-w-[150px] truncate">
                          <a href={`/experiments/${exp.id}`} className="text-blue-400 hover:underline font-mono text-xs">
                            {exp.id.slice(0, 12)}
                          </a>
                        </td>
                        <td className="py-2 text-right font-mono text-xs text-cyan-400">
                          {(details.val_acc as number)?.toFixed(4) ?? '—'}
                        </td>
                        <td className="py-2 text-right font-mono text-xs text-emerald-400">
                          {(details.test_acc_id as number)?.toFixed(4) ?? '—'}
                        </td>
                        <td className="py-2 text-right font-mono text-sm">
                          {(details.ood_f1 as number)?.toFixed(4) ?? '—'}
                        </td>
                        <td className="py-2 text-right text-xs text-slate-400">
                          {details.elapsed_s ? `${details.elapsed_s}s` : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-slate-400">No runs yet.</p>
          )}
        </div>

        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Auto-Loop Experiments</h2>
          {loopExps.length > 0 ? (
            <div className="space-y-4">
              {loopExps.slice(0, 3).map((loop) => {
                const iterCount = loop.iterations?.length ?? 0;
                const keptCount = loop.iterations?.filter(i => i.status === 'keep').length ?? 0;
                const bestAcc = loop.iterations
                  ? Math.max(...loop.iterations.map(i => i.test_acc_id ?? 0))
                  : 0;
                return (
                  <div key={loop.id} className="bg-slate-900 rounded p-4 border border-slate-700">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <a href={`/experiments/${loop.id}`} className="text-purple-400 hover:underline font-mono text-sm">
                          {loop.id}
                        </a>
                        <p className="text-xs text-slate-500 mt-0.5">
                          {iterCount} iterations ({keptCount} kept)
                        </p>
                      </div>
                      <span className="text-sm font-bold text-emerald-400">
                        {bestAcc.toFixed(4)}
                      </span>
                    </div>
                    <ScoreChart scores={loop.scores || []} color="#8b5cf6" height={100} />
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-slate-400">No auto-loop experiments yet.</p>
          )}
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-xl font-semibold mb-3">About This Project</h2>
        <div className="text-sm text-slate-300 space-y-2 leading-relaxed">
          <p>
            <strong className="text-slate-100">Task:</strong> Train a classifier on <strong className="text-cyan-400">PneumoniaMNIST</strong> (2 classes: normal, pneumonia),
            then detect whether a chest X-ray from <strong className="text-emerald-400">ChestMNIST</strong> is normal, pneumonia, or an <strong className="text-purple-400">unseen OOD class</strong> (consolidation).
            This simulates a real-world medical scenario where a model must flag novel diseases it was never trained on.
          </p>
          <p>
            <strong className="text-slate-100">Metrics:</strong>
          </p>
          <ul className="list-disc list-inside space-y-1 pl-2">
            <li><span className="text-cyan-400 font-mono">Val Acc</span> — accuracy on PneumoniaMNIST validation (same distribution as training). Shows how well the model learns the training classes.</li>
            <li><span className="text-emerald-400 font-mono">ID Test Acc</span> — accuracy on ChestMNIST normal + pneumonia only (domain-shifted but same classes). Reveals generalization gap across datasets.</li>
            <li><span className="text-purple-400 font-mono">OOD F1</span> — F1 score for detecting consolidation as out-of-distribution. Measures how well the model flags unseen classes.</li>
          </ul>
          <p>
            <strong className="text-slate-100">Architecture:</strong> SimpleCNN (2 conv layers + 2 fc layers) with 3-class output, LeakyReLU, and dropout.
            The OOD detection uses a softmax confidence threshold (default 0.7): if max probability is below threshold, the sample is labeled as OOD.
          </p>
          <p className="text-slate-500 text-xs pt-1">
            MLSS26_HACKATHON — Scientific AutoResearch loop for chest X-ray OOD detection.
          </p>
        </div>
      </div>
    </div>
  );
}
