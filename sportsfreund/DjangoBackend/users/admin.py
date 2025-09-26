from django.contrib import admin
from .models import User, Exercise, Session

admin.site.register(User)
admin.site.register(Exercise)
admin.site.register(Session)

