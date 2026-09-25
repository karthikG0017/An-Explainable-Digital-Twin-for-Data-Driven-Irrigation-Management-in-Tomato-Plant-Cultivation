/**
 * App.jsx — Digital Twin Dashboard
 *
 * Item 7a: Live Status View
 * Item 7d: What-If Simulator
 * Shows two plant cards side-by-side (Plant A: fixed-schedule, Plant B: AI-managed).
 * Fetches status via API (mock or live) and displays all FusionResult fields.
 */
import { useState, useEffect, useCallback } from 'react';
import PlantCard from './components/PlantCard';
import WhatIfSimulator from './components/WhatIfSimulator';
import { fetchPlantStatus } from './api';
import './index.css';

export default function App() {
  const [plantA, setPlantA] = useState({ data: null, error: null, loading: true });
  const [plantB, setPlantB] = useState({ data: null, error: null, loading: true });
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAll = useCallback(async () => {
    setRefreshing(true);
    setPlantA((prev) => ({ ...prev, loading: true, error: null }));
    setPlantB((prev) => ({ ...prev, loading: true, error: null }));

    // Fetch both plants concurrently
    const [resA, resB] = await Promise.allSettled([
      fetchPlantStatus('A'),
      fetchPlantStatus('B'),
    ]);

    setPlantA({
      data: resA.status === 'fulfilled' ? resA.value : null,
      error: resA.status === 'rejected' ? resA.reason.message : null,
      loading: false,
    });

    setPlantB({
      data: resB.status === 'fulfilled' ? resB.value : null,
      error: resB.status === 'rejected' ? resB.reason.message : null,
      loading: false,
    });

    setLastUpdated(new Date());
    setRefreshing(false);
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const formatTime = (date) => {
    if (!date) return '—';
    return date.toLocaleTimeString('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  };

  return (
    <>
      <header className="app-header">
        <h1>Digital Twin — Tomato Irrigation Dashboard</h1>
        <p>Explainable AI-driven irrigation management · Two-plant A/B comparison</p>
      </header>

      <main className="dashboard-container">
        {/* Refresh bar */}
        <div className="refresh-bar">
          <span className="last-updated">
            Last updated: {formatTime(lastUpdated)}
          </span>
          <button
            className="refresh-btn"
            onClick={fetchAll}
            disabled={refreshing}
          >
            {refreshing ? '⟳ Refreshing...' : '⟳ Refresh'}
          </button>
        </div>

        {/* Two-plant comparison grid */}
        <div className="plants-grid">
          <PlantCard
            plantId="A"
            label="Plant A"
            typeLabel="Fixed-schedule irrigation (control)"
            data={plantA.data}
            error={plantA.error}
            loading={plantA.loading}
          />
          <PlantCard
            plantId="B"
            label="Plant B"
            typeLabel="AI-managed irrigation (experimental)"
            data={plantB.data}
            error={plantB.error}
            loading={plantB.loading}
          />
        </div>

        {/* Item 7d: What-If Simulator */}
        <WhatIfSimulator />
      </main>
    </>
  );
}
