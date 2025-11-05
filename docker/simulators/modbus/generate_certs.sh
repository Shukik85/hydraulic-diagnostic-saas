#!/bin/bash

# SSL Certificate Generation for Modbus TCP Server
# Fixed for Git Bash on Windows compatibility

set -e

CERTS_DIR="./certs"
CONFIG_FILE="./openssl.cnf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔐 SSL Certificate Generation for Modbus TCP${NC}"
echo -e "${BLUE}=================================================${NC}"

# Create certificates directory
if [ ! -d "$CERTS_DIR" ]; then
    mkdir -p "$CERTS_DIR"
    echo -e "${GREEN}✅ Created certificates directory: $CERTS_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  Certificates directory already exists${NC}"
fi

# Create OpenSSL configuration file
cat > "$CONFIG_FILE" << 'EOL'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = RU
ST = Moscow
L = Moscow
O = Hydraulic Diagnostic Platform
OU = Development Team
CN = modbus-server

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = modbus-server
DNS.3 = modbus-sim
IP.1 = 127.0.0.1
IP.2 = 172.17.0.2
IP.3 = 192.168.1.100
EOL

echo -e "${GREEN}✅ Created OpenSSL configuration${NC}"

# Generate private key (simple RSA, no PKCS8 complications)
echo -e "${BLUE}🔑 Generating RSA private key...${NC}"
openssl genrsa -out "$CERTS_DIR/server.key" 2048

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Private key generated: $CERTS_DIR/server.key${NC}"
else
    echo -e "${RED}❌ Failed to generate private key${NC}"
    exit 1
fi

# Generate certificate signing request (using config file to avoid Git Bash path issues)
echo -e "${BLUE}📝 Generating certificate signing request...${NC}"
openssl req -new -key "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.csr" -config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ CSR generated: $CERTS_DIR/server.csr${NC}"
else
    echo -e "${RED}❌ Failed to generate CSR${NC}"
    exit 1
fi

# Generate self-signed certificate (1 year validity)
echo -e "${BLUE}🏆 Generating self-signed certificate...${NC}"
openssl x509 -req -in "$CERTS_DIR/server.csr" -signkey "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.crt" -days 365 -extensions v3_req -extfile "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Certificate generated: $CERTS_DIR/server.crt${NC}"
else
    echo -e "${RED}❌ Failed to generate certificate${NC}"
    exit 1
fi

# Set proper permissions (important for security)
chmod 600 "$CERTS_DIR/server.key"
chmod 644 "$CERTS_DIR/server.crt"

echo -e "${GREEN}🔒 Certificate permissions set${NC}"

# Cleanup temporary files
rm -f "$CONFIG_FILE" "$CERTS_DIR/server.csr"

# Show certificate details
echo -e "${GREEN}🎉 SSL certificates generated successfully!${NC}"
echo -e "${BLUE}📋 Certificate details:${NC}"
openssl x509 -in "$CERTS_DIR/server.crt" -text -noout | grep -A5 "Subject:"
openssl x509 -in "$CERTS_DIR/server.crt" -text -noout | grep -A10 "X509v3 Subject Alternative Name:"

echo -e "\n${BLUE}📁 Generated files:${NC}"
ls -la "$CERTS_DIR/"

echo -e "\n${YELLOW}⚡ Next steps:${NC}"
echo -e "${YELLOW}   1. Verify server_config.json has TLS enabled${NC}"
echo -e "${YELLOW}   2. Start Modbus server with TLS: docker-compose up -d${NC}"
echo -e "${YELLOW}   3. Test secure connection${NC}"

echo -e "\n${GREEN}✅ Ready for secure Modbus TCP communication!${NC}"