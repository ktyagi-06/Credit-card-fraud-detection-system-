const defaultConfig = {
  main_title: "Advanced Fraud Detection Dashboard",
  subtitle: "Real-time ML-Powered Transaction Monitoring & Analysis",
  analyze_button: "🔍 Analyze Transaction",
  generate_data_button: "🎲 Generate Random",
  background_color: "#0f0c29",
  surface_color: "rgba(255, 255, 255, 0.05)",
  text_color: "#ffffff",
  primary_action_color: "#667eea",
  secondary_action_color: "#764ba2"
};

let transactionCount = 0;
let fraudCount = 0;
let transactions = [];

// Synthetic data generation (mimicking Python's numpy/pandas)
function generateSyntheticTransaction() {
  const normalRandom = () => {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  return {
    amount: Math.abs(normalRandom() * 500 + 250),
    time: Math.floor(Math.random() * 86400),
    v1: normalRandom() * 2,
    v2: normalRandom() * 2,
    v3: normalRandom() * 2,
    v4: normalRandom() * 2
  };
}

// ML prediction algorithm (simulating sklearn/xgboost)
function mlPredict(amount, time, v1, v2, v3, v4) {
  const features = [
    amount / 1000,
    Math.sin(time / 10000),
    v1, v2, v3, v4
  ];

  let score = 0;
  const weights = [0.3, 0.15, 0.15, 0.15, 0.15, 0.1];
  
  features.forEach((feature, i) => {
    score += Math.abs(feature) * weights[i];
  });

  score += Math.random() * 0.2;
  
  const isFraud = score > 0.65;
  const confidence = isFraud ? 
    Math.min(score * 100, 99) : 
    Math.min((1 - score) * 100, 99);

  return {
    isFraud,
    confidence: confidence.toFixed(1),
    riskScore: (score * 10).toFixed(2),
    anomalyIndex: (Math.abs(v1 + v2 + v3 + v4) / 4).toFixed(2)
  };
}

function updateStats() {
  document.getElementById('totalTransactions').textContent = transactionCount;
  document.getElementById('fraudDetected').textContent = fraudCount;
  document.getElementById('transactionChange').textContent = 
    ((transactionCount / Math.max(transactionCount - 10, 1)) * 100 - 100).toFixed(1) + '%';
  document.getElementById('fraudChange').textContent = 
    ((fraudCount / Math.max(fraudCount - 2, 1)) * 100 - 100).toFixed(1) + '%';
}

function addTransaction(amount, isFraud, confidence) {
  transactionCount++;
  if (isFraud) fraudCount++;

  const transaction = {
    id: transactionCount,
    amount: amount,
    isFraud: isFraud,
    confidence: confidence,
    timestamp: new Date().toLocaleTimeString()
  };

  transactions.unshift(transaction);
  if (transactions.length > 10) transactions.pop();

  const transactionList = document.getElementById('transactionList');
  transactionList.innerHTML = '';

  transactions.forEach(t => {
    const item = document.createElement('div');
    item.className = 'transaction-item';
    item.innerHTML = `
      <div class="transaction-info">
        <div class="transaction-amount">$${t.amount.toFixed(2)}</div>
        <div class="transaction-details">ID: #${t.id} • ${t.timestamp} • Confidence: ${t.confidence}%</div>
      </div>
      <div class="transaction-status ${t.isFraud ? 'fraud' : 'safe'}">
        ${t.isFraud ? '⚠️ FRAUD' : '✓ SAFE'}
      </div>
    `;
    transactionList.appendChild(item);
  });

  updateStats();
  updateChart();
}

function updateChart() {
  const chartBars = document.getElementById('chartBars');
  chartBars.innerHTML = '';

  const hours = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'];
  const data = hours.map(() => Math.floor(Math.random() * 100) + 20);

  data.forEach((value, i) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = value + '%';
    bar.innerHTML = `
      <div class="chart-bar-value">${value}</div>
      <div class="chart-bar-label">${hours[i]}</div>
    `;
    chartBars.appendChild(bar);
  });

  // Update model chart
  const modelChart = document.getElementById('modelChart');
  modelChart.innerHTML = '';
  const models = ['XGBoost', 'RF', 'NN'];
  const accuracies = [98.7, 97.2, 96.8];

  accuracies.forEach((acc, i) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = acc + '%';
    bar.innerHTML = `
      <div class="chart-bar-value">${acc}%</div>
      <div class="chart-bar-label">${models[i]}</div>
    `;
    modelChart.appendChild(bar);
  });
}

function updateProgressRing(percent) {
  const circle = document.getElementById('progressCircle');
  const radius = circle.r.baseVal.value;
  const circumference = radius * 2 * Math.PI;
  
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  circle.style.strokeDashoffset = circumference;
  
  setTimeout(() => {
    const offset = circumference - (percent / 100) * circumference;
    circle.style.strokeDashoffset = offset;
  }, 100);
}

document.getElementById('generateBtn').addEventListener('click', function(e) {
  e.preventDefault();
  const data = generateSyntheticTransaction();
  
  document.getElementById('amount').value = data.amount.toFixed(2);
  document.getElementById('time').value = data.time;
  document.getElementById('v1').value = data.v1.toFixed(2);
  document.getElementById('v2').value = data.v2.toFixed(2);
  document.getElementById('v3').value = data.v3.toFixed(2);
  document.getElementById('v4').value = data.v4.toFixed(2);
});

document.getElementById('analyzeBtn').addEventListener('click', async function(e) {
  e.preventDefault();

  const amount = parseFloat(document.getElementById('amount').value) || 0;
  const time = parseFloat(document.getElementById('time').value) || 0;
  const v1 = parseFloat(document.getElementById('v1').value) || 0;
  const v2 = parseFloat(document.getElementById('v2').value) || 0;
  const v3 = parseFloat(document.getElementById('v3').value) || 0;
  const v4 = parseFloat(document.getElementById('v4').value) || 0;

  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.disabled = true;

  document.getElementById('predictionResult').classList.remove('show');
  document.getElementById('loading').classList.add('show');

  await new Promise(resolve => setTimeout(resolve, 1500));

  const prediction = mlPredict(amount, time, v1, v2, v3, v4);

  document.getElementById('loading').classList.remove('show');

  const predictionIcon = document.getElementById('predictionIcon');
  const predictionText = document.getElementById('predictionText');
  const predictionConfidence = document.getElementById('predictionConfidence');
  const progressCircle = document.getElementById('progressCircle');

  if (prediction.isFraud) {
    predictionIcon.textContent = '⚠️';
    predictionText.textContent = 'FRAUD DETECTED';
    predictionText.style.color = '#f87171';
    progressCircle.setAttribute('stroke', '#f87171');
  } else {
    predictionIcon.textContent = '✅';
    predictionText.textContent = 'Transaction Approved';
    predictionText.style.color = '#4ade80';
    progressCircle.setAttribute('stroke', '#4ade80');
  }

  predictionConfidence.textContent = `Confidence: ${prediction.confidence}%`;
  updateProgressRing(parseFloat(prediction.confidence));

  document.getElementById('predictionResult').classList.add('show');

  addTransaction(amount, prediction.isFraud, prediction.confidence);

  analyzeBtn.disabled = false;
});

// Initialize with some data
updateChart();

// Generate initial synthetic transactions
for (let i = 0; i < 5; i++) {
  const data = generateSyntheticTransaction();
  const prediction = mlPredict(data.amount, data.time, data.v1, data.v2, data.v3, data.v4);
  addTransaction(data.amount, prediction.isFraud, prediction.confidence);
}

async function onConfigChange(config) {
  const mainTitle = config.main_title || defaultConfig.main_title;
  const subtitle = config.subtitle || defaultConfig.subtitle;
  const analyzeButton = config.analyze_button || defaultConfig.analyze_button;
  const generateDataButton = config.generate_data_button || defaultConfig.generate_data_button;
  const backgroundColor = config.background_color || defaultConfig.background_color;
  const textColor = config.text_color || defaultConfig.text_color;

  document.getElementById('main-title').textContent = mainTitle;
  document.getElementById('subtitle').textContent = subtitle;
  document.getElementById('analyze-button-text').textContent = analyzeButton;
  document.getElementById('generate-data-button-text').textContent = generateDataButton;

  document.body.style.color = textColor;
}

if (window.elementSdk) {
  window.elementSdk.init({
    defaultConfig,
    onConfigChange,
    mapToCapabilities: (config) => ({
      recolorables: [
        {
          get: () => config.background_color || defaultConfig.background_color,
          set: (value) => {
            config.background_color = value;
            window.elementSdk.setConfig({ background_color: value });
          }
        },
        {
          get: () => config.text_color || defaultConfig.text_color,
          set: (value) => {
            config.text_color = value;
            window.elementSdk.setConfig({ text_color: value });
          }
        },
        {
          get: () => config.primary_action_color || defaultConfig.primary_action_color,
          set: (value) => {
            config.primary_action_color = value;
            window.elementSdk.setConfig({ primary_action_color: value });
          }
        }
      ],
      borderables: [],
      fontEditable: undefined,
      fontSizeable: undefined
    }),
    mapToEditPanelValues: (config) => new Map([
      ["main_title", config.main_title || defaultConfig.main_title],
      ["subtitle", config.subtitle || defaultConfig.subtitle],
      ["analyze_button", config.analyze_button || defaultConfig.analyze_button],
      ["generate_data_button", config.generate_data_button || defaultConfig.generate_data_button]
    ])
  });
}
