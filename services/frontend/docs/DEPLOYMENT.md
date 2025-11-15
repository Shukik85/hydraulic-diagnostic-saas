# 🚀 Deployment Guide

> Production deployment instructions

---

## 📦 Docker Deployment

### Build Image

```bash
docker build -t hydraulic-frontend:latest .
```

### Run Container

```bash
docker run -p 3000:3000 \
  -e NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com \
  -e NUXT_PUBLIC_ENABLE_RAG=true \
  hydraulic-frontend:latest
```

---

## ⚙️ Environment Variables

**Production `.env`:**
```bash
NUXT_PUBLIC_ENVIRONMENT=production
NUXT_PUBLIC_API_BASE=https://api.hydraulic-diagnostics.com/api/v1
NUXT_PUBLIC_WS_BASE=wss://api.hydraulic-diagnostics.com/ws
NUXT_PUBLIC_ENABLE_RAG=true
NUXT_PUBLIC_FORCE_HTTPS=true
```

---

## ✅ Production Checklist

- [ ] Environment variables configured
- [ ] API client generated
- [ ] Type check passes
- [ ] Build succeeds
- [ ] SSL certificates configured
- [ ] CORS configured on backend
- [ ] Monitoring enabled

---

**Last Updated:** November 15, 2025