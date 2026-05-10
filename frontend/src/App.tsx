import { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts';
import './App.css';

interface Prediction {
  prediction_class: number;
  probabilities: {
    Down: number;
    Flat: number;
    Up: number;
  };
  confidence: number;
  trade_signal: string;
  threshold: number;
}

interface ChartData {
  time: string;
  price: number;
  prediction: Prediction;
}

interface PredictionData {
  ticker: string;
  last_price: number;
  timestamp: string;
  chart_data: ChartData[];
  prediction: Prediction;
}

function App() {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<ChartData | null>(null);
  const [animationKey, setAnimationKey] = useState<number>(0);
  const [period, setPeriod] = useState<number>(50);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('http://127.0.0.1:8000/api/latest');
      setData(response.data);
      if (response.data.chart_data && response.data.chart_data.length > 0) {
        setSelectedPoint(response.data.chart_data[response.data.chart_data.length - 1]);
      }
    } catch (err: any) {
      setError(err.message || 'Error connecting to the FastAPI backend. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  const handleChartClick = (state: any) => {
    if (state && state.activePayload && state.activePayload.length > 0) {
      setSelectedPoint(state.activePayload[0].payload);
      setAnimationKey(Date.now());
    }
  };

  if (loading) {
    return <div className="loading-container"><h2>Loading Live Market Data...</h2><div className="spinner"></div></div>;
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={fetchData}>Retry</button>
      </div>
    );
  }

  if (!data || !selectedPoint) return null;

  const { prediction } = selectedPoint;
  const isTradeSignal = prediction.trade_signal !== 'HOLD';
  const signalColor = prediction.trade_signal.includes('BUY') ? '#22c55e' : prediction.trade_signal.includes('SELL') ? '#ef4444' : '#64748b';
  const isHistorical = selectedPoint.time !== data.timestamp;

  const displayChartData = data.chart_data.slice(-period);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Gold Price Direction Predictor</h1>
        <p className="subtitle">Trading Instrument: <strong>{data.ticker}</strong> Futures</p>
      </header>

      <main className="main-content">
        <section className="chart-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2>Market Overview</h2>
            <select 
              value={period} 
              onChange={(e) => setPeriod(Number(e.target.value))}
              style={{ padding: '0.5rem', borderRadius: '4px', backgroundColor: '#334155', color: '#f8fafc', border: 'none', cursor: 'pointer' }}
            >
              <option value={50}>Last 50 Hours</option>
              <option value={100}>Last 100 Hours</option>
              <option value={200}>Last 200 Hours</option>
              <option value={720}>Last 1 Month</option>
            </select>
          </div>
          
          <div className="price-info">
            <span className="current-price">${data.last_price.toFixed(2)}</span>
            <span className="timestamp">Latest: {new Date(data.timestamp).toLocaleString()}</span>
          </div>
          
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={displayChartData} onClick={handleChartClick} style={{ cursor: 'crosshair' }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="time" 
                  tickFormatter={(tick) => {
                    const d = new Date(tick);
                    return period > 100 ? `${d.getDate()}/${d.getMonth()+1}` : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                  }}
                  stroke="#888"
                />
                <YAxis domain={['auto', 'auto']} stroke="#888" />
                <Tooltip 
                  labelFormatter={(label) => new Date(label).toLocaleString()}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                />
                <Line type="monotone" dataKey="price" stroke="#fbbf24" strokeWidth={2} dot={false} activeDot={{ r: 8, fill: '#fbbf24' }} isAnimationActive={false} />
                {selectedPoint && (
                  <ReferenceDot x={selectedPoint.time} y={selectedPoint.price} r={7} fill="#22c55e" stroke="#fff" strokeWidth={3} isFront={true} />
                )}
              </LineChart>
            </ResponsiveContainer>
            <p className="chart-hint">💡 Hover to see the yellow dot, then CLICK EXACTLY ON THE LINE to see the model's prediction at that moment.</p>
          </div>
        </section>

        <section key={`section-${animationKey}`} className="prediction-section fade-in">
          <h2>
            AI Prediction Analysis
            {isHistorical && <span className="historical-badge">Historical Point</span>}
          </h2>
          <p className="selected-time">Analyzing data for: <strong>{new Date(selectedPoint.time).toLocaleString()}</strong></p>
          
          <div className="metrics-grid">
            <div className="metric-card spin-card" style={{ borderTop: `4px solid ${signalColor}` }}>
              <h3>Trade Signal</h3>
              <div className="metric-value" style={{ color: signalColor }}>{prediction.trade_signal}</div>
              <p className="metric-desc">Based on {prediction.threshold * 100}% threshold logic</p>
            </div>
            
            <div className="metric-card spin-card">
              <h3>Model Confidence</h3>
              <div className="metric-value">{(prediction.confidence * 100).toFixed(1)}%</div>
              <p className="metric-desc">Certainty of the top predicted class</p>
            </div>
            
            <div className="metric-card spin-card">
              <h3>Top Probability</h3>
              <div className="metric-value">
                {prediction.prediction_class === 0 ? "Down" : prediction.prediction_class === 2 ? "Up" : "Flat"}
              </div>
              <p className="metric-desc">Highest likelihood outcome</p>
            </div>
          </div>

          <div className="probabilities-bar">
            <h3>Class Probabilities</h3>
            <div className="progress-container">
              <div className="progress-bar down" style={{ width: `${prediction.probabilities.Down * 100}%` }}>
                {(prediction.probabilities.Down * 100).toFixed(1)}%
              </div>
              <div className="progress-bar flat" style={{ width: `${prediction.probabilities.Flat * 100}%` }}>
                {(prediction.probabilities.Flat * 100).toFixed(1)}%
              </div>
              <div className="progress-bar up" style={{ width: `${prediction.probabilities.Up * 100}%` }}>
                {(prediction.probabilities.Up * 100).toFixed(1)}%
              </div>
            </div>
            <div className="legend">
              <span><span className="dot down-dot"></span> Down</span>
              <span><span className="dot flat-dot"></span> Flat</span>
              <span><span className="dot up-dot"></span> Up</span>
            </div>
          </div>

          <div className="explanation-box">
            <h3>How it works</h3>
            <p>
              This Stacking Ensemble model analyzes 96 features including intermarket correlations (Silver, Oil, DXY) and momentum.
              It uses a selective trading strategy: it only issues a <strong>BUY</strong> or <strong>SELL</strong> signal if its confidence 
              exceeds the strictly optimized threshold of <strong>{(prediction.threshold * 100).toFixed(0)}%</strong>.
            </p>
            <p>
              {isTradeSignal 
                ? `At this point in time, the model was ${(prediction.confidence * 100).toFixed(1)}% confident, which is ABOVE the threshold. Therefore, a trade signal was issued.` 
                : `At this point in time, the model was only ${(prediction.confidence * 100).toFixed(1)}% confident, which is BELOW the threshold. The signal was HOLD to avoid unnecessary risk.`}
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
