from django.urls import path
<<<<<<< HEAD
from . import views

urlpatterns = [
    path('register/', views.UserRegistrationView.as_view(), name='user-register'),
    path('login/', views.UserLoginView.as_view(), name='user-login'),
    path('logout/', views.user_logout, name='user-logout'),
    path('profile/', views.UserProfileView.as_view(), name='user-profile'),
    path('profile/extended/', views.UserProfileExtendedView.as_view(), name='user-profile-extended'),
    path('change-password/', views.ChangePasswordView.as_view(), name='change-password'),
    path('activity/', views.UserActivityView.as_view(), name='user-activity'),
    path('stats/', views.UserStatsView.as_view(), name='user-stats'),
=======
from rest_framework_simplejwt.views import TokenRefreshView
from .views import CustomTokenObtainPairView, UserRegistrationView, user_profile

urlpatterns = [
    path('auth/login/', CustomTokenObtainPairView.as_view(), name='login'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', UserRegistrationView.as_view(), name='register'),
    path('auth/profile/', user_profile, name='profile'),
>>>>>>> cae71f2baa2fcddf341336d7eaa5721b089eeb9f
]
