from django.db import models

# User
class User(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)

    def __str__(self):
        return self.name


# Exercise
class Exercise(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    DIFFICULTY_CHOICES = [
        ('E', 'Easy'),
        ('M', 'Medium'),
        ('H', 'Hard'),
    ]
    difficulty = models.CharField(max_length=1, choices=DIFFICULTY_CHOICES)

    def __str__(self):
        return self.name


# Session
class Session(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    exercises = models.ManyToManyField(Exercise, related_name='sessions')
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    total_score = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Session {self.id} of {self.user.name}"
