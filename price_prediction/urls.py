from django.urls import path,include
from . import views
from rest_framework.routers import DefaultRouter
from .views import HistoricalPriceViewSet

router = DefaultRouter()
router.register(r'PredictionModel', HistoricalPriceViewSet, basename='predictionmodel')

urlpatterns = [
    path('home/', views.home, name='home'),
    path('', views.index, name='index'),
    path('api/', include(router.urls)),
    path('api/serve_overall_table/', views.serve_overall_table, name='serve_overall_table'),
    path('account/', views.account, name='account'),
    path('login/', views.user_login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('home/commodity_detail/', views.commodity_detail, name='commodity_detail'),
    path('train_model/', views.train_model, name='train_model'),
    path('potato/', views.potato_detail, name='potato'),

  ]