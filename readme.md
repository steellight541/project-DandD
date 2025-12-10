# Idea
i have 2 ideas for the project; first one is a DnD master where i make a program where you can play a DnD game with a AI as the master, second idea is having 2 AIs, talk to each other, and have a conversation about a topic, and see how the conversation goes.

# Progress
## DnD master
### brainstorming
- models
    - [Ollama/magistral (tools, thinking, 24b) ](https://ollama.com/library/magistral)
    - [Ollama/magistral-abliterated (thinking, 24b)](https://ollama.com/huihui_ai/magistral-abliterated)
    - [Ollama/openhermes (7.5b)]()
    - [Ollama/mistral-nemo-30b]

# deps
- transformers
- lightrag-hku


## cuda versioning to allow GPU 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install lightrag-hku
```