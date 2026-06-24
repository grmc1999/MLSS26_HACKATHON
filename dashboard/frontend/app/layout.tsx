import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Flu Experiment Dashboard',
  description: 'Autonomous research pipeline dashboard for ILI forecasting',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100">
        <nav className="border-b border-slate-800 px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <a href="/" className="text-xl font-bold text-amber-400">
              🤒 Flu Dashboard
            </a>
            <div className="flex gap-6 text-sm">
              <a href="/" className="hover:text-amber-400 transition">Overview</a>
              <a href="/experiments" className="hover:text-amber-400 transition">Experiments</a>
              <a href="/leaderboard" className="hover:text-amber-400 transition">Leaderboard</a>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
