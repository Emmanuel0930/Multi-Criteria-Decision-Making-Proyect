from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
print('GET /health ->', client.get('/health').status_code)
print('GET /health JSON ->', client.get('/health').json())
print('\nGET /models ->', client.get('/models').status_code)
print('GET /models JSON ->', client.get('/models').json())
