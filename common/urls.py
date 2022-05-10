from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "common"

urlpatterns = [
    path('', auth_views.LoginView.as_view(template_name='common/firstmain.html'), name='firstmain'),
    path('login/', auth_views.LoginView.as_view(template_name='common/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('mainpage/', auth_views.LoginView.as_view(template_name='common/mainpage.html'), name='mainpage'),
    path('signup/', views.signup, name='signup'),
]