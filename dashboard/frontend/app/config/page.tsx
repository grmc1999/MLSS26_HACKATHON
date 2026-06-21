'use client';

import { useState, useEffect } from 'react';

interface Agent {
  name: string;
  display_name: string;
  model: string;
  upgrade_model: string;
}

interface ModelInfo {
  id: string;
  name: string;
  context_length: number;
  modalities: string[];
}

interface ModelsData {
  free_models: ModelInfo[];
  premium_models: ModelInfo[];
}

export default function ConfigPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [models, setModels] = useState<ModelsData | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    async function fetchData() {
      const agentsRes = await fetch('/api/agents');
      const agentsData = await agentsRes.json();
      setAgents(agentsData.agents || []);

      const modelsRes = await fetch('/api/models');
      setModels(await modelsRes.json());
    }
    fetchData();
  }, []);

  async function swapModel() {
    if (!selectedAgent || !selectedModel) return;
    try {
      const res = await fetch(`/api/agents/${selectedAgent}/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: selectedModel }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setMessage(`Successfully swapped ${selectedAgent} to ${selectedModel}`);
        const agentsRes = await fetch('/api/agents');
        const agentsData = await agentsRes.json();
        setAgents(agentsData.agents || []);
      }
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Model Configuration</h1>
      <p className="text-slate-400">Swap LLM models for each agent. All free models cost $0 via OpenRouter.</p>

      {message && (
        <div className="bg-blue-900 text-blue-200 px-4 py-3 rounded">{message}</div>
      )}

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Swap Agent Model</h2>
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm text-slate-400 mb-1">Agent</label>
            <select
              value={selectedAgent ?? ''}
              onChange={(e) => {
                setSelectedAgent(e.target.value);
                const agent = agents.find((a) => a.name === e.target.value);
                setSelectedModel(agent?.model ?? '');
              }}
              className="w-full bg-slate-900 text-slate-100 px-4 py-2 rounded border border-slate-700"
            >
              <option value="">Select an agent...</option>
              {agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.display_name} ({a.name})
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-sm text-slate-400 mb-1">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-900 text-slate-100 px-4 py-2 rounded border border-slate-700"
            >
              <option value="">Select a model...</option>
              <optgroup label="Free Models ($0)">
                {models?.free_models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.id})
                  </option>
                ))}
              </optgroup>
              <optgroup label="Premium Models (paid)">
                {models?.premium_models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.id})
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
          <button
            onClick={swapModel}
            disabled={!selectedAgent || !selectedModel}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 px-6 py-2 rounded font-semibold"
          >
            Apply
          </button>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Current Agent Models</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-2">Agent</th>
              <th className="text-left py-2">Current Model</th>
              <th className="text-left py-2">Upgrade Path</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a) => (
              <tr key={a.name} className="border-b border-slate-800">
                <td className="py-2">{a.display_name}</td>
                <td className="py-2"><code className="text-green-400">{a.model}</code></td>
                <td className="py-2"><code className="text-yellow-400">{a.upgrade_model}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
