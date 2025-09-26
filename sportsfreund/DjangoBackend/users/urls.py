from rest_framework.routers import DefaultRouter
from .views import UserViewSet, ExerciseViewSet, SessionViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'exercises', ExerciseViewSet)
router.register(r'sessions', SessionViewSet)

urlpatterns = router.urls
