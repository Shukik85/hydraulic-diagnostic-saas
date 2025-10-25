export default defineEventHandler(async () => {
  // In a real app, fetch precomputed JSON from a cache or service.
  // For demo, we embed the latest 7-day arrays derived from BTC volatility.
  return {
    sparklines: {
      temperature: [54.3, 55.0, 53.1, 53.1, 52.2, 54.0, 52.2],
      pressure: [146.1, 145.0, 147.9, 147.8, 149.1, 146.5, 149.2],
      flow_rate: [98.1, 100.0, 95.2, 95.3, 93.1, 97.5, 93.0],
      vibration: [0.84, 1.23, 2.12, 1.21, 1.85, 1.90, 2.47]
    }
  }
})
