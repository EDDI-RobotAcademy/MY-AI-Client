import ollama

response = ollama.chat(model='gemma2:2b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])