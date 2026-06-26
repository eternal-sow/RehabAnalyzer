import lmstudio as lms

# Без указания токена — SDK будет работать без аутентификации
model = lms.llm()
response = model.respond("Привет!")
print(response.content)