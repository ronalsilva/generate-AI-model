from transformers import pipeline

# Inicializa a pipeline com o modelo especificado
pipe = pipeline("text-generation", model="ninho86/teste")

# Gera texto a partir de uma entrada de exemplo
input_text = "Mona Lisa"
generated_text = pipe(input_text, max_length=50, num_return_sequences=1)

# Exibe o texto gerado
print(generated_text[0]['generated_text'])