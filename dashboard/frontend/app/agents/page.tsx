'use client';

import { useState, useEffect } from 'react';

interface Agent {
  name: string;
  display_name: string;
  description: string;
  model: string;
  fast_model: string;
  upgrade_model: string;
  skills: string[];
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);

  useEffect(() => {
    async function fetchAgents() {
      const res = await fetch('/api/agents');
      const data = await res.json();
      setAgents(data.agents || []);
    }
    fetchAgents();
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Agents</h1>
      <p className="text-slate-400">8 specialized agents powered by free OpenRouter models</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {agents.map((agent) => (
          <div key={agent.name} className="bg-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-lg font-semibold">{agent.display_name}</h2>
              <code className="text-xs text-blue-400 bg-slate-900 px-2 py-1 rounded">
                {agent.name}
              </code>
            </div>
            <p className="text-sm text-slate-400 mb-4">{agent.description}</p>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-slate-500">Model: </span>
                <code className="text-green-400">{agent.model}</code>
              </div>
              <div>
                <span className="text-slate-500">Fast Model: </span>
                <code className="text-green-400">{agent.fast_model}</code>
              </div>
              <div>
                <span className="text-slate-500">Upgrade: </span>
                <code className="text-yellow-400">{agent.upgrade_model}</code>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {agent.skills.map((skill) => (
                <span key={skill} className="px-2 py-1 bg-slate-700 rounded text-xs text-slate-300">
                  {skill}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
