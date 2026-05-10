import { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './App.css';

interface ChartData {
  time: string;
  price: number;
}

interface PredictionData {
  ticker: string;
  last_price: number;
  timestamp: string;
  chart_data: ChartData[];
  prediction: {
    prediction_class: number;
    probabilities: {
      Down: number;
      Flat: number;
      Up: number;
    };
    confidence: number;
    trade_signal: string;
    threshold: number;
  };
}

function App() {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('http://127.0.0.1:8000/api/latest');
      setData(response.data);
    } catch (err: any) {
      setError(err.message || 'Error connecting to the FastAPI backend. Is it running?');
    } finally {
      setLoading(false);
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

  if (!data) return null;

  const { prediction } = data;
  const isTradeSignal = prediction.trade_signal !== 'HOLD';
  const signalColor = prediction.trade_signal.includes('BUY') ? '#22c55e' : prediction.trade_signal.includes('SELL') ? '#ef4444' : '#64748b';

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Gold Price Direction Predictor</h1>
        <p className="subtitle">Trading Instrument: <strong>{data.ticker}</strong> Futures</p>
      </header>

      <main className="main-content">
        <section className="chart-section">
          <h2>Market Overview (Last 50 Hours)</h2>
          <div className="price-info">
            <span className="current-price">${data.last_price.toFixed(2)}</span>
            <span className="timestamp">Last Updated: {new Date(data.timestamp).toLocaleString()}</span>
          </div>
          
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.chart_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="time" 
                  tickFormatter={(tick) => new Date(tick).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  stroke="#888"
                />
                <YAxis domain={['auto', 'auto']} stroke="#888" />
                <Tooltip 
                  labelFormatter={(label) => new Date(label).toLocaleString()}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                />
                <Line type="monotone" dataKey="price" stroke="#fbbf24" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="prediction-section">
          <h2>AI Prediction Analysis</h2>
          
          <div className="metrics-grid">
            <div className="metric-card" style={{ borderTop: `4px solid ${signalColor}` }}>
              <h3>Trade Signal</h3>
              <div className="metric-value" style={{ color: signalColor }}>{prediction.trade_signal}</div>
              <p className="metric-desc">Based on {prediction.threshold * 100}% threshold logic</p>
            </div>
            
            <div className="metric-card">
              <h3>Model Confidence</h3>
              <div className="metric-value">{(prediction.confidence * 100).toFixed(1)}%</div>
              <p className="metric-desc">Certainty of the top predicted class</p>
            </div>
            
            <div className="metric-card">
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
            <h3>📖 How it works</h3>
            <p>
              This Stacking Ensemble model analyzes 96 features including intermarket correlations (Silver, Oil, DXY) and momentum.
              It uses a selective trading strategy: it only issues a <strong>BUY</strong> or <strong>SELL</strong> signal if its confidence 
              exceeds the strictly optimized threshold of <strong>{(prediction.threshold * 100).toFixed(0)}%</strong>.
            </p>
            <p>
              {isTradeSignal 
                ? `Currently, the model is ${(prediction.confidence * 100).toFixed(1)}% confident, which is ABOVE the threshold. Therefore, a trade signal is issued.` 
                : `Currently, the model is only ${(prediction.confidence * 100).toFixed(1)}% confident, which is BELOW the threshold. The signal is HOLD to avoid unnecessary risk.`}
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
