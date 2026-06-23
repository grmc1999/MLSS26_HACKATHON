'use client';

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}

function hashIndex(id: string, max: number): number {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = id.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % max;
}

export function ScoreChart({
  scores,
  color,
  height = 300,
}: {
  scores: { step: number; score: number }[];
  color?: string;
  height?: number;
}) {
  if (!scores || scores.length === 0) return <p className="text-slate-400 py-8 text-center">No score data</p>;
  const lineColor = color || '#2563eb';
  return (
    <div style={{ width: '100%', height }}>
      <svg viewBox={`0 0 1000 ${height}`} className="w-full h-full" preserveAspectRatio="none">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
          <line key={frac} x1="0" y1={height * (1 - frac)} x2="1000" y2={height * (1 - frac)}
                stroke="#334155" strokeWidth="1" />
        ))}
        {scores.length > 1 && (
          <>
            {(() => {
              const minStep = scores[0].step;
              const maxStep = scores[scores.length - 1].step;
              const stepRange = Math.max(maxStep - minStep, 1);
              const minScore = Math.min(...scores.map(s => s.score));
              const maxScore = Math.max(...scores.map(s => s.score));
              const scoreRange = Math.max(maxScore - minScore, 0.001);
              const points = scores.map(s =>
                `${((s.step - minStep) / stepRange) * 1000},${height * (1 - (s.score - minScore) / scoreRange)}`
              ).join(' ');
              return <polyline fill="none" stroke={lineColor} strokeWidth="2.5" points={points} />;
            })()}
          </>
        )}
      </svg>
      <div className="flex justify-between text-xs text-slate-500 mt-1">
        <span>Step {scores[0]?.step ?? 0}</span>
        <span>Step {scores[scores.length - 1]?.step ?? 0}</span>
      </div>
    </div>
  );
}

export function StatCard({ label, value, sub, color }: StatCardProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-5 border border-slate-700" style={color ? { borderLeftColor: color, borderLeftWidth: 3 } : undefined}>
      <p className="text-sm text-slate-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color ? '' : 'text-white'}`}
         style={color ? { color } : undefined}>
        {value}
      </p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}

export function SourceBadge({ source }: { source?: string }) {
  const styles: Record<string, string> = {
    logs: 'bg-blue-900/50 text-blue-300 border-blue-700',
    run_exp: 'bg-emerald-900/50 text-emerald-300 border-emerald-700',
    auto_loop: 'bg-purple-900/50 text-purple-300 border-purple-700',
  };
  const labels: Record<string, string> = {
    logs: 'MLAgent',
    run_exp: 'Run',
    auto_loop: 'AutoLoop',
  };
  const style = styles[source || 'logs'] || styles.logs;
  const label = labels[source || 'logs'] || source;
  return (
    <span className={`text-xs px-2 py-0.5 rounded border ${style}`}>
      {label}
    </span>
  );
}

export function StatusBadge({ status }: { status?: string }) {
  const styles: Record<string, string> = {
    completed: 'bg-green-900/50 text-green-300 border-green-700',
    running: 'bg-yellow-900/50 text-yellow-300 border-yellow-700',
    pending: 'bg-slate-700/50 text-slate-400 border-slate-600',
    unknown: 'bg-slate-700/50 text-slate-400 border-slate-600',
  };
  const style = styles[status || 'unknown'] || styles.unknown;
  return (
    <span className={`text-xs px-2 py-0.5 rounded border ${style}`}>
      {status || 'unknown'}
    </span>
  );
}

const EXPERIMENT_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

export function getExpColor(id: string, index?: number): string {
  if (index !== undefined) return EXPERIMENT_COLORS[index % EXPERIMENT_COLORS.length];
  return EXPERIMENT_COLORS[hashIndex(id, EXPERIMENT_COLORS.length)];
}
