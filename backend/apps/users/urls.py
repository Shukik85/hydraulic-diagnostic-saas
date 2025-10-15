from django.urls import path
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
]
