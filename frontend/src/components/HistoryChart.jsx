/**
 * HistoryChart.jsx — Historical sensor trend chart for a single plant.
 *
 * Item 7c: Renders soil moisture, temperature, and humidity over time
 * using Chart.js. Shows the 70% irrigation threshold as a horizontal
 * reference line on the moisture scale.
 *
 * Uses the last 200 data points (~4 days at 30-min intervals) for readability.
 */
import { useEffect, useState, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { fetchHistory } from '../api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const IRRIGATION_THRESHOLD = 70;
const DATA_POINTS = 200; // last ~4 days

export default function HistoryChart({ plantId }) {
  const [histData, setHistData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const chartRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchHistory(plantId, DATA_POINTS)
      .then((res) => {
        if (!cancelled) {
          setHistData(res.data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [plantId]);

  if (loading) {
    return (
      <div className="history-chart-container">
        <div className="history-chart-title">Historical Trends</div>
        <div className="history-chart-loading">Loading historical data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="history-chart-container">
        <div className="history-chart-title">Historical Trends</div>
        <div className="history-chart-error">⚠️ {error}</div>
      </div>
    );
  }

  if (!histData || histData.length === 0) {
    return (
      <div className="history-chart-container">
        <div className="history-chart-title">Historical Trends</div>
        <div className="history-chart-loading">No historical data available</div>
      </div>
    );
  }

  // Format timestamps for x-axis: show "Jun 15 12:00" style
  const labels = histData.map((d) => {
    const dt = new Date(d.timestamp);
    const month = dt.toLocaleString('en-US', { month: 'short' });
    const day = dt.getDate();
    const hours = dt.getHours().toString().padStart(2, '0');
    const mins = dt.getMinutes().toString().padStart(2, '0');
    return `${month} ${day} ${hours}:${mins}`;
  });

  // Threshold line — same length as data
  const thresholdLine = histData.map(() => IRRIGATION_THRESHOLD);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Soil Moisture (%)',
        data: histData.map((d) => d.soil_moisture_pct),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        yAxisID: 'y',
        order: 1,
      },
      {
        label: 'Temperature (°C)',
        data: histData.map((d) => d.temperature_c),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        yAxisID: 'y',
        order: 2,
      },
      {
        label: 'Humidity (%)',
        data: histData.map((d) => d.humidity_pct),
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        yAxisID: 'y',
        order: 3,
      },
      {
        label: 'Irrigation Threshold (70%)',
        data: thresholdLine,
        borderColor: 'rgba(239, 68, 68, 0.5)',
        borderWidth: 1.5,
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false,
        yAxisID: 'y',
        order: 0,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          usePointStyle: true,
          pointStyle: 'line',
          boxWidth: 24,
          padding: 12,
          font: { family: 'Inter', size: 11 },
          color: '#5a5f72',
        },
      },
      tooltip: {
        backgroundColor: '#1a1d26',
        titleFont: { family: 'Inter', size: 12 },
        bodyFont: { family: 'Inter', size: 11 },
        padding: 10,
        cornerRadius: 6,
        callbacks: {
          label: (ctx) => {
            const val = ctx.parsed.y;
            if (ctx.dataset.label.includes('Threshold')) return null;
            if (ctx.dataset.label.includes('Temperature')) {
              return ` ${ctx.dataset.label}: ${val.toFixed(1)}°C`;
            }
            return ` ${ctx.dataset.label}: ${val.toFixed(1)}%`;
          },
        },
      },
      title: { display: false },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 8,
          maxRotation: 0,
          font: { family: 'Inter', size: 10 },
          color: '#8b90a0',
        },
        grid: { display: false },
      },
      y: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20,
          font: { family: 'Inter', size: 10 },
          color: '#8b90a0',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
  };

  return (
    <div className="history-chart-container">
      <div className="history-chart-title">
        Historical Trends
        <span className="history-chart-subtitle">
          Last {histData.length} readings (~{Math.round(histData.length * 0.5 / 24)} days)
        </span>
      </div>
      <div className="history-chart-wrapper">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
      </div>
    </div>
  );
}
