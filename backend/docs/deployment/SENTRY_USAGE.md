# Sentry usage guide

## 1) Create a project at sentry.io
- Project type: Python / Django
- Copy DSN from Settings → Client Keys (DSN)

## 2) Add environment variables
- In backend/.env.production or server env:
```
SENTRY_DSN=your-dsn-here
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

## 3) How it works in this repo
- We initialize Sentry in `backend/core/sentry_setup.py`
- `backend/core/settings.py` imports it at startup (add this if missing):
```
# At the end of settings.py
try:
    from .sentry_setup import *  # noqa
except Exception:
    pass
```

## 4) What gets captured
- Unhandled exceptions from Django views and DRF
- Celery task failures
- Optional tracing if `SENTRY_TRACES_SAMPLE_RATE > 0`
- No PII (send_default_pii=False)

## 5) Verify Sentry works
- Trigger a test exception locally:
```
from django.http import HttpResponse

def boom(request):
    1/0
    return HttpResponse('ok')
```
- Add to urls.py and hit the endpoint
- Check event appears in Sentry

## 6) Best practices
- Set different environments: development / staging / production
- Use releases: `SENTRY_RELEASE=1.0.0` for version pinning
- Add breadcrumbs in critical flows
- Use performance tracing for slow endpoints selectively

## 7) Troubleshooting
- No events: check DSN and outbound network
- Flood of events: add ignore rules / sample rate
- Sensitive data: confirm `send_default_pii=False`
