# Support Management Module

**Система управления тикетами поддержки с SLA tracking и восстановлением доступа.**

## 🎯 Возможности

### Support Tickets
- ✅ Автоматическая генерация номеров тикетов (TKT-YYYY-XXXXX)
- ✅ Категории: Technical, Billing, Access, Feature, Bug
- ✅ Приоритеты с SLA: Low (72h), Medium (24h), High (8h), Critical (2h)
- ✅ Статусы workflow: New → Open → In Progress → Resolved → Closed
- ✅ SLA tracking с автоматическим breach detection
- ✅ Threaded conversations (messages)
- ✅ Auto-assignment к доступным агентам

### Access Recovery
- ✅ Password Reset
- ✅ 2FA Reset
- ✅ Account Unlock
- ✅ Email Change
- ✅ Workflow с верификацией

### Email Notifications
- ✅ Ticket created/updated/resolved
- ✅ SLA warnings и breach alerts
- ✅ Assignment notifications
- ✅ HTML templates

### Django Admin
- ✅ Rich UI с SLA indicators
- ✅ Bulk actions (assign, resolve, escalate)
- ✅ Inline message threading
- ✅ Search & filters
- ✅ Color-coded badges

### Celery Tasks
- ✅ Асинхронные email уведомления
- ✅ Periodic SLA monitoring
- ✅ Auto-assignment logic

---

## 📦 Models

### SupportTicket

```python
from apps.support.models import SupportTicket

ticket = SupportTicket.objects.create(
    user=user,
    category=SupportTicket.Category.TECHNICAL,
    priority=SupportTicket.Priority.HIGH,
    subject="Cannot login",
    description="Getting 500 error",
)

# Auto-generated
print(ticket.ticket_number)  # TKT-2025-00123
print(ticket.sla_due_date)   # 8 hours from now
```

### TicketMessage

```python
from apps.support.models import TicketMessage

message = TicketMessage.objects.create(
    ticket=ticket,
    author=support_agent,
    message="Investigating the issue...",
)
```

### AccessRecoveryRequest

```python
from apps.support.models import AccessRecoveryRequest

request = AccessRecoveryRequest.objects.create(
    user=user,
    request_type=AccessRecoveryRequest.RequestType.PASSWORD_RESET,
)

# Approve/reject
request.approve(admin, notes="Verified via email")
request.reject(admin, reason="Failed verification")
```

---

## 🔧 Celery Tasks

### Send Notification

```python
from apps.support.tasks import send_ticket_notification

# Queue email
send_ticket_notification.delay(
    ticket_id=str(ticket.id),
    notification_type="created",
)
```

### SLA Monitoring (Periodic)

```python
from apps.support.tasks import check_sla_breaches

# Runs every 30 minutes via Celery Beat
result = check_sla_breaches()
print(result)
# {'approaching_sla': 3, 'breached_sla': 1}
```

### Auto-Assignment

```python
from apps.support.tasks import auto_assign_tickets

# Assign unassigned tickets
result = auto_assign_tickets()
print(result)
# {'assigned': 5, 'total_agents': 3}
```

---

## 🔐 Permissions

**Доступ к Django Admin**:
- `is_staff=True` для просмотра
- `is_superuser=True` для полного управления

**Bulk actions** требуют staff permissions.

---

## ⚙️ Configuration

### Settings

```python
# config/settings.py

INSTALLED_APPS = [
    # ...
    'apps.support',
]

# Frontend URL for email links
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Email configuration
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
DEFAULT_FROM_EMAIL = 'support@hydraulic-diagnostics.com'
```

### Celery Beat Schedule

```python
# config/settings.py

from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'check-sla-breaches': {
        'task': 'apps.support.tasks.check_sla_breaches',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },
    'auto-assign-tickets': {
        'task': 'apps.support.tasks.auto_assign_tickets',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
}
```

---

## 🧪 Testing

```bash
# Run tests
pytest apps/support/tests/ -v

# With coverage
pytest apps/support/tests/ --cov=apps.support --cov-report=term-missing
```

---

## 📊 Admin Interface

### Access

```
http://localhost:8000/admin/support/
```

### Features

**SupportTicket Admin**:
- Color-coded badges (category, priority, status)
- SLA indicators with countdown
- Bulk actions: assign, resolve, escalate
- Inline message thread
- Search by ticket number, subject, user

**AccessRecoveryRequest Admin**:
- Approve/reject actions
- Verification method tracking
- Admin notes

---

## 🚀 Usage Examples

### Creating a Ticket

```python
ticket = SupportTicket.objects.create(
    user=request.user,
    category=SupportTicket.Category.BILLING,
    priority=SupportTicket.Priority.MEDIUM,
    subject="Invoice question",
    description="Need clarification on last invoice",
)

# Send notification
send_ticket_notification.delay(str(ticket.id), "created")
```

### Adding a Message

```python
message = TicketMessage.objects.create(
    ticket=ticket,
    author=request.user,
    message="Here's more info...",
)

# Notify assigned agent
if ticket.assigned_to:
    send_ticket_notification.delay(str(ticket.id), "updated")
```

### Checking SLA Status

```python
if ticket.is_overdue:
    print("⚠️ SLA BREACHED!")
elif ticket.time_until_sla:
    print(f"Time left: {ticket.time_until_sla}")
```

---

## 🔗 Integration

### With User Model

```python
# Get user's tickets
tickets = request.user.support_tickets.all()

# Get assigned tickets (for staff)
assigned = request.user.assigned_tickets.filter(status='open')
```

### With Notifications Module

```python
# If you have notifications app
from apps.notifications.models import Notification

Notification.objects.create(
    user=ticket.user,
    title=f"Ticket {ticket.ticket_number} updated",
    message=ticket.subject,
    link=f"/support/{ticket.ticket_number}",
)
```

---

## 📈 Metrics

### SLA Compliance

```python
from django.db.models import Count, Q

stats = SupportTicket.objects.filter(
    status__in=['resolved', 'closed'],
    created_at__gte=last_month,
).aggregate(
    total=Count('id'),
    sla_met=Count('id', filter=Q(sla_breached=False)),
)

compliance_rate = stats['sla_met'] / stats['total'] * 100
print(f"SLA Compliance: {compliance_rate:.1f}%")
```

### Average Resolution Time

```python
from django.db.models import Avg, F

avg_time = SupportTicket.objects.filter(
    status='resolved',
).aggregate(
    avg_resolution=Avg(F('resolved_at') - F('created_at'))
)

print(f"Avg resolution: {avg_time['avg_resolution']}")
```

---

## 🐛 Troubleshooting

### Emails not sending

```bash
# Check Celery worker
celery -A config worker -l info

# Check email settings
python manage.py shell
>>> from django.core.mail import send_mail
>>> send_mail('Test', 'Body', 'from@example.com', ['to@example.com'])
```

### SLA not updating

```bash
# Check Celery beat
celery -A config beat -l info

# Manually run task
python manage.py shell
>>> from apps.support.tasks import check_sla_breaches
>>> check_sla_breaches()
```

---

## 📝 TODO / Future Enhancements

- [ ] Ticket templates для быстрых ответов
- [ ] Customer satisfaction rating
- [ ] Knowledge base integration
- [ ] Multi-language support
- [ ] File attachments
- [ ] Ticket merging
- [ ] Canned responses
- [ ] Ticket tags/labels

---

**Built with ❤️ for Hydraulic Diagnostics SaaS**
