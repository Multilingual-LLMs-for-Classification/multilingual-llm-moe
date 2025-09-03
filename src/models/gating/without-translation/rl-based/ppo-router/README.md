## Usage Instructions:

### Basic Usage:

```
system = PromptRoutingSystem()
result = system.route_prompt("Analyze Tesla stock sentiment")
```

### With RL Training:

```
training_data = create_sample_training_data()
system.train_agents(training_data, epochs=50)
result = system.route_prompt("Your prompt", use_rl=True)
```

### Batch Processing:

```
prompts = ["prompt1", "prompt2", "prompt3"]
results = system.batch_process(prompts)\
```

### Save logs while running the program

```
"y" | python -X utf8 .\ppo-router.py *> result_all_lang.log
```
