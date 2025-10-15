from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

urlpatterns = [
    path('auth/login/', views.CustomTokenObtainPairView.as_view(), name='login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', views.UserRegistrationView.as_view(), name='register'),
    path('auth/profile/', views.user_profile, name='profile'),
    path('logout/', views.user_logout, name='user-logout'),
    path('profile/extended/', views.UserProfileExtendedView.as_view(), name='user-profile-extended'),
    path('change-password/', views.ChangePasswordView.as_view(), name='change-password'),
    path('activity/', views.UserActivityView.as_view(), name='user-activity'),
    path('stats/', views.UserStatsView.as_view(), name='user-stats'),
]
