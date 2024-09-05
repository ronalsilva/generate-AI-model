import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Carregar modelo pré-treinado e tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Definir o token de preenchimento
tokenizer.pad_token = tokenizer.eos_token

# Ler dados de treinamento do arquivo JSON
with open('data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Preparar dados para treinamento
inputs = [item["input"] for item in train_data]
outputs = [item["output"] for item in train_data]

# Tokenizar os dados
train_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=128)
labels_encodings = tokenizer(outputs, truncation=True, padding=True, max_length=128)

# Criar Dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

train_dataset = SimpleDataset(train_encodings, labels_encodings)

# Data collator para garantir que os lotes tenham o mesmo tamanho
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Definir parâmetros de treinamento
training_args = TrainingArguments(
    output_dir='E:/workspace/IA/results',  # Alterar para um local com mais espaço
    num_train_epochs=1,
    per_device_train_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='E:/workspace/IA/logs',  # Alterar para um local com mais espaço
)

# Treinar o modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# Salvar o modelo ajustado
model.save_pretrained("E:/workspace/IA/results")  # Alterar para um local com mais espaço
tokenizer.save_pretrained("E:/workspace/IA/results")  # Alterar para um local com mais espaço