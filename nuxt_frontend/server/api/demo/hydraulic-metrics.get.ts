export default defineEventHandler(async (event) => {
  // Get query parameters
  const query = getQuery(event)
  const source = (query.source as string) || 'btc'
  
  // Mock data sources
  const sources = {
    btc: {
      key: 'HYD-001',
      name: 'Pump Station A',
      sparklines: {
        temperature: [65, 67, 64, 68, 66, 69, 67, 65, 63, 64, 66, 68],
        pressure: [2.1, 2.3, 2.2, 2.4, 2.3, 2.1, 2.2, 2.3, 2.4, 2.2, 2.1, 2.3],
        flow_rate: [180, 175, 185, 190, 188, 182, 179, 183, 186, 184, 181, 187],
        vibration: [0.8, 0.9, 0.7, 1.0, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.9]
      },
      thresholds: {
        temperature: { green: 70, red: 80 },
        pressure: { green: 2.5, red: 3.0 },
        flow_rate: { green: 200, red: 250 },
        vibration: { green: 1.0, red: 1.5 }
      }
    },
    eth: {
      key: 'HYD-002',
      name: 'Hydraulic Motor B',
      sparklines: {
        temperature: [70, 72, 69, 73, 71, 74, 72, 70, 68, 69, 71, 73],
        pressure: [1.8, 2.0, 1.9, 2.1, 2.0, 1.8, 1.9, 2.0, 2.1, 1.9, 1.8, 2.0],
        flow_rate: [150, 145, 155, 160, 158, 152, 149, 153, 156, 154, 151, 157],
        vibration: [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5, 0.6, 0.7, 0.6, 0.5, 0.7]
      },
      thresholds: {
        temperature: { green: 75, red: 85 },
        pressure: { green: 2.2, red: 2.7 },
        flow_rate: { green: 170, red: 200 },
        vibration: { green: 0.8, red: 1.2 }
      }
    }
  }
  
  // Safe selection with fallback
  const selected = sources[source as keyof typeof sources] || sources.btc
  
  // Helper function with readonly array handling
  const avg = (xs: readonly number[]): number => {
    const arr = [...xs] // Convert readonly to mutable
    return Math.round((arr.reduce((a: number, b: number) => a + b, 0) / arr.length) * 100) / 100
  }
  
  return {
    system: {
      key: selected.key,
      name: selected.name,
      health_score: Math.floor(Math.random() * 20) + 80,
      status: 'active',
      last_update: new Date().toISOString()
    },
    metrics: {
      temperature: {
        current: selected.sparklines.temperature[selected.sparklines.temperature.length - 1],
        average: avg(selected.sparklines.temperature),
        threshold_green: selected.thresholds.temperature.green,
        threshold_red: selected.thresholds.temperature.red,
        sparkline: [...selected.sparklines.temperature]
      },
      pressure: {
        current: selected.sparklines.pressure[selected.sparklines.pressure.length - 1],
        average: avg(selected.sparklines.pressure),
        threshold_green: selected.thresholds.pressure.green,
        threshold_red: selected.thresholds.pressure.red,
        sparkline: [...selected.sparklines.pressure]
      },
      flow_rate: {
        current: selected.sparklines.flow_rate[selected.sparklines.flow_rate.length - 1],
        average: avg(selected.sparklines.flow_rate),
        threshold_green: selected.thresholds.flow_rate.green,
        threshold_red: selected.thresholds.flow_rate.red,
        sparkline: [...selected.sparklines.flow_rate]
      },
      vibration: {
        current: selected.sparklines.vibration[selected.sparklines.vibration.length - 1],
        average: avg(selected.sparklines.vibration),
        threshold_green: selected.thresholds.vibration.green,
        threshold_red: selected.thresholds.vibration.red,
        sparkline: [...selected.sparklines.vibration]
      }
    }
  }
})