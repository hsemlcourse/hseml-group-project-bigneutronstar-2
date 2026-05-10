import { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import type { InteractionItem } from 'chart.js';
import { Line, getElementAtEvent } from 'react-chartjs-2';
import './App.css';

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

interface ChartDataObj {
  time: string;
  price: number;
  prediction: Prediction;
}

interface PredictionData {
  ticker: string;
  last_price: number;
  timestamp: string;
  chart_data: ChartDataObj[];
  prediction: Prediction;
}

function App() {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  const [selectedPoint, setSelectedPoint] = useState<ChartDataObj | null>(null);
  const [animationKey, setAnimationKey] = useState<number>(0);
  const [period, setPeriod] = useState<number>(50);

  const chartRef = useRef<any>(null);

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

  const handleChartClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!chartRef.current || !data) return;
    const elements: InteractionItem[] = getElementAtEvent(chartRef.current, event);
    
    if (elements.length > 0) {
      const dataIndex = elements[0].index;
      const displayData = data.chart_data.slice(-period);
      const clickedPoint = displayData[dataIndex];
      
      setSelectedPoint(clickedPoint);
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

  const chartDataConfig = {
    labels: displayChartData.map(d => {
      const date = new Date(d.time);
      return period > 100 
        ? `${date.getDate()}/${date.getMonth()+1} ${date.getHours()}:00`
        : `${date.getHours().toString().padStart(2, '0')}:00`;
    }),
    datasets: [
      {
        label: 'Price',
        data: displayChartData.map(d => d.price),
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251, 191, 36, 0.1)',
        borderWidth: 2,
        pointRadius: displayChartData.map(d => d.time === selectedPoint.time ? 8 : 0),
        pointBackgroundColor: displayChartData.map(d => d.time === selectedPoint.time ? '#22c55e' : '#fbbf24'),
        pointBorderColor: displayChartData.map(d => d.time === selectedPoint.time ? '#ffffff' : '#fbbf24'),
        pointBorderWidth: displayChartData.map(d => d.time === selectedPoint.time ? 3 : 0),
        pointHoverRadius: 6,
        fill: true,
        tension: 0.1
      }
    ]
  };

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f8fafc',
        bodyColor: '#f8fafc',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
        displayColors: false,
        callbacks: {
          label: function(context: any) {
            return `Price: $${context.parsed.y.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: '#334155',
          drawBorder: false,
        },
        ticks: {
          color: '#888',
          maxTicksLimit: 10
        }
      },
      y: {
        grid: {
          color: '#334155',
          drawBorder: false,
        },
        ticks: {
          color: '#888',
        }
      }
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Gold Price Direction Predictor</h1>
        <p className="subtitle">Trading Instrument: <strong>{data.ticker}</strong> Futures</p>
      </header>

      <main className="main-content">
        <section className="chart-section">
          <div className="controls-row">
            <h2>Market Overview</h2>
            <div className="period-selector">
              <label htmlFor="period-select">Period: </label>
              <select 
                id="period-select"
                value={period} 
                onChange={(e) => setPeriod(Number(e.target.value))}
              >
                <option value={50}>Last 50 Hours</option>
                <option value={100}>Last 100 Hours</option>
                <option value={200}>Last 200 Hours</option>
                <option value={720}>Last 1 Month</option>
              </select>
            </div>
          </div>
          
          <div className="price-info">
            <span className="current-price">${data.last_price.toFixed(2)}</span>
            <span className="timestamp">Latest: {new Date(data.timestamp).toLocaleString()}</span>
          </div>
          
          <div className="chart-container" style={{ height: '350px', cursor: 'crosshair' }}>
            <Line 
              ref={chartRef}
              data={chartDataConfig} 
              options={chartOptions} 
              onClick={handleChartClick} 
            />
          </div>
          <p className="chart-hint">💡 Hover and CLICK EXACTLY ON THE CHART AREA to select a point in time.</p>
        </section>

        <section className="prediction-section">
          <h2>
            AI Prediction Analysis
            {isHistorical && <span className="historical-badge">Historical Point</span>}
          </h2>
          <p className="selected-time">Analyzing data for: <strong>{new Date(selectedPoint.time).toLocaleString()}</strong></p>
          
          {/* Key forces React to destroy and recreate these elements, guaranteeing the animation runs */}
          <div className="metrics-grid" key={`grid-${animationKey}`}>
            <div className="metric-card spin-card" style={{ borderTop: `4px solid ${signalColor}` }}>
              <h3>Trade Signal</h3>
              <div className="metric-value" style={{ color: signalColor }}>{prediction.trade_signal}</div>
              <p className="metric-desc">Based on {prediction.threshold * 100}% threshold logic</p>
            </div>
            
            <div className="metric-card spin-card" style={{ animationDelay: '0.1s' }}>
              <h3>Model Confidence</h3>
              <div className="metric-value">{(prediction.confidence * 100).toFixed(1)}%</div>
              <p className="metric-desc">Certainty of the top predicted class</p>
            </div>
            
            <div className="metric-card spin-card" style={{ animationDelay: '0.2s' }}>
              <h3>Top Probability</h3>
              <div className="metric-value">
                {prediction.prediction_class === 0 ? "Down" : prediction.prediction_class === 2 ? "Up" : "Flat"}
              </div>
              <p className="metric-desc">Highest likelihood outcome</p>
            </div>
          </div>

          <div className="probabilities-bar" key={`bar-${animationKey}`}>
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
