# Modbus TCP Simulator for Development

**Purpose:** Test Modbus TCP protocol handlers without real industrial equipment.

## 🚀 Quick Start

### Option 1: baetyltech/modbus-simulator (your choice)
```bash
# Try the image you found
docker run -d --name modbus-sim \
  -p 1502:502 \
  baetyltech/modbus-simulator:latest
```

### Option 2: oitc/modbus-server (fallback)
```bash
# Alternative if baetyltech doesn't work
docker run -d --name modbus-sim \
  -p 1502:502 \
  oitc/modbus-server:latest
```

### Option 3: Custom Python Simulator (most reliable)
```bash
# Create your own using pymodbus
python modbus_server_local.py
```

## 🔧 Test Connection

1. **Update test script configuration:**
   ```python
   # In test_modbus_simple.py
   HOST = "127.0.0.1"
   PORT = 1502
   UNIT_ID = 1
   ```

2. **Run test:**
   ```bash
   python test_modbus_simple.py
   ```

## 📊 Expected Results

**Successful connection:**
```
✅ pymodbus imported successfully
🚀 Starting Modbus TCP Test...
📦 pymodbus available: True
📡 Testing connection to 127.0.0.1:1502 (Unit ID: 1)
✅ Connected to Modbus TCP device at 127.0.0.1:1502
✅ CONNECTION SUCCESSFUL!
```

**Register reading:**
```
📊 Testing Register 0 (address 0, type uint16)...
✅ Raw registers: [0]
✅ Decoded value: 0 (uint16)

📊 Testing Holding Register 40001 (address 40001, type uint16)...
✅ Raw registers: [1234]
✅ Decoded value: 1234 (uint16)
```

## 🐛 Troubleshooting

### Container Issues:
```bash
# Check if container is running
docker ps

# Check container logs
docker logs modbus-sim

# Remove and retry
docker rm -f modbus-sim
```

### Connection Issues:
```bash
# Check port is open
netstat -ano | findstr 1502

# Test raw TCP connection
telnet 127.0.0.1 1502
```

### Image Not Found Solutions:

1. **Try alternative images:**
   - `oitc/modbus-server:latest`
   - `digitalpetri/modbus-slave-tcp`
   - Build custom from pymodbus

2. **Use local Python server:**
   ```bash
   pip install pymodbus>=3.6.6
   python modbus_server_local.py
   ```

## 📝 Simulator Configuration

### Default Values:
- **Port:** 502 (mapped to host 1502)
- **Unit ID:** 1
- **Registers:** Usually initialized to 0

### Custom Values (depends on image):
Some simulators support environment variables:
```bash
-e MODBUS_UNIT_ID=1 \
-e HOLDING_REGISTER_0=1234 \
-e HOLDING_REGISTER_1=42
```

## 🎯 Next Steps After Successful Test

1. **Integrate with Django sensors app**
2. **Create Celery task for periodic data collection** 
3. **Store readings in TimescaleDB**
4. **Connect to ML service for anomaly detection**
5. **Display real-time data in UI**

---

**💡 Tip:** If all Docker images fail, the Python local server (Option 3) is the most reliable for development!
