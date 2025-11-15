"""
Monitoring views
"""
from django.db import connection
from django.http import JsonResponse


def health_check(request):
    """Health check endpoint for Docker"""
    try:
        # Check database connection
        connection.ensure_connection()
        return JsonResponse({'status': 'ok', 'database': 'connected'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
