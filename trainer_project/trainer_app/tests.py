from django.test import TestCase
from .models import *


class UserModelTest(TestCase):

    @classmethod
    def setUpTestData(cls):
        Authentication.objects.create(id=1, email="test@gmail.com", password="password")

    def test_token(self):
        user = Authentication.objects.get(id=1)
        self.assertEquals(user.token, None)

    def test_email_max_length(self):
        user = Authentication.objects.get(id=1)
        max_length = user._meta.get_field("email").max_length
        self.assertEquals(max_length, 100)

    def test_password_max_length(self):
        user = Authentication.objects.get(id=1)
        max_length = user._meta.get_field("password").max_length
        self.assertEquals(max_length, 100)


class ViewTest(TestCase):

    @classmethod
    def setUpTestData(cls):
        pass

    def test_register_view(self):
        resp = self.client.post('/register/', data={"email": "test@gmail.com", "password": "pass"})
        self.assertEqual(resp.status_code, 200)
        users = Authentication.objects.filter(email="test@gmail.com")
        self.assertEqual(users.count(), 1)
        self.assertEqual(users[0].password, "pass")

    def test_login_view(self):
        self.test_register_view()
        resp = self.client.post('/login/', data={"email": "test@gmail.com", "password": "pass"})
        self.assertEqual(resp.status_code, 200)
        users = Authentication.objects.filter(email="test@gmail.com")
        self.assertEqual(users.count(), 1)
        self.assertEqual(users[0].password, "pass")

    def test_logout_view(self):
        self.test_register_view()
        token = Authentication.objects.get(email="test@gmail.com").token
        resp = self.client.post('/logout/', data={"token": token})
        self.assertEqual(resp.status_code, 200)
        user = Authentication.objects.get(email="test@gmail.com")
        self.assertEqual(user.token, None)
        self.assertEqual(resp.content.decode(), "ok")

    def test_password_update_view(self):
        self.test_register_view()
        user = Authentication.objects.get(email="test@gmail.com")
        self.assertEqual(user.password, "pass")
        resp = self.client.post('/password_update/', data={"email": "test@gmail.com", "password": "new_pass"})
        self.assertEqual(resp.status_code, 200)
        user = Authentication.objects.get(email="test@gmail.com")
        self.assertEqual(user.password, "new_pass")
