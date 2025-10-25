export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const source = (query.source || 'btc').toString()

  // BTC-derived (как было) + ETH-derived (новое)
  const btc = {
    key: 'btc',
    name: 'Bitcoin (Нестабильная)',
    sparklines: {
      temperature: [54.3, 55.0, 53.1, 53.1, 52.2, 54.0, 52.2],
      pressure: [146.1, 145.0, 147.9, 147.8, 149.1, 146.5, 149.2],
      flow_rate: [98.1, 100.0, 95.2, 95.3, 93.1, 97.5, 93.0],
      vibration: [0.84, 1.23, 2.12, 1.21, 1.85, 1.90, 2.47]
    },
    thresholds: {
      temperature: { green: 50.0, red: 54.0 },
      pressure: { green: 148.0, red: 152.0 },
      flow_rate: { green: 90.0, red: 96.0 },
      vibration: { green: 1.2, red: 2.0 }
    }
  }

  const eth = {
    key: 'eth',
    name: 'Ethereum (Экстремальная)',
    sparklines: {
      temperature: [53.0, 55.0, 53.4, 52.5, 50.5, 50.7, 48.7],
      pressure: [151.0, 148.0, 150.4, 151.8, 154.8, 154.4, 157.4],
      flow_rate: [90.8, 95.0, 91.5, 89.6, 85.4, 85.9, 81.7],
      vibration: [0.9, 1.97, 1.87, 1.31, 2.61, 1.16, 2.7]
    },
    thresholds: {
      temperature: { green: 48.75, red: 52.65 },
      pressure: { green: 157.35, red: 151.5 },
      flow_rate: { green: 81.75, red: 90.05 },
      vibration: { green: 1.19, red: 1.94 }
    }
  }

  const sources = { btc, eth }
  const selected = sources[source] || btc

  // Aggregates for the selected source (min/max/avg for a week)
  const arr = selected.sparklines
  const avg = (xs) => Math.round((xs.reduce((a, b) => a + b, 0) / xs.length) * 100) / 100
  const week_stats = {
    temperature: { min: Math.min(...arr.temperature), max: Math.max(...arr.temperature), avg: avg(arr.temperature) },
    pressure: { min: Math.min(...arr.pressure), max: Math.max(...arr.pressure), avg: avg(arr.pressure) },
    flow_rate: { min: Math.min(...arr.flow_rate), max: Math.max(...arr.flow_rate), avg: avg(arr.flow_rate) },
    vibration: { min: Math.min(...arr.vibration), max: Math.max(...arr.vibration), avg: avg(arr.vibration) }
  }

  return {
    source: selected.key,
    name: selected.name,
    sparklines: selected.sparklines,
    thresholds: selected.thresholds,
    aggregates: { week_stats }
  }
})
