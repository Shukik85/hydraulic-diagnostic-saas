# apps/users/urls.py

from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    CustomTokenObtainPairView,
    UserActivityViewSet,
    UserProfileViewSet,
    UserRegistrationView,
    UserViewSet,
)

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="users")
router.register(r"profile", UserProfileViewSet, basename="profile")
router.register(r"activity", UserActivityViewSet, basename="activity")

urlpatterns = [
    path("auth/login/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/register/", UserRegistrationView.as_view(), name="user-register"),
    path("", include(router.urls)),
]
