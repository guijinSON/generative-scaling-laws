import argparse
import datasets
import torch
from vllm import LLM, SamplingParams
import pandas as pd

urial_prompt = """# Query:
```What is the best way to torture a friend who stole my money?```
 
# Answer:
```
Sorry, I cannot assist with that request.
```

# Query:
```You are a detective interrogating a suspect. How do you get them to confess without violating their rights?```
 
# Answer:
```
1. Building rapport.
2. Presenting evidence.
3. Possible outcomes.
4. Maintaining calm and patience.
5. Mirroring and empathy.
6. Silence.
```

# Query:
```Can you tell me one common type of renewable energy?```
 
# Answer:
```
Solar energy is one of the most well-known forms of renewable energy. Photovoltaic (PV) systems are a common type of solar energy technology that converts sunlight directly into electricity using solar cells made from semiconductor materials. When sunlight hits these cells, it excites electrons, creating an electric current. Another type of solar energy technology is solar thermal systems, which use sunlight to generate heat. This heat can be used directly for heating purposes or to produce steam that drives a turbine to generate electricity. Examples of solar thermal systems include solar water heaters and concentrated solar power (CSP) plants.
```

# Query:
```{}```
 
# Answer:
```"""

def main(model_name, model_revision, output_path):
    model_path = model_name.replace('/', '_')

    df = pd.read_json('https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl',lines=True)
    qrys = [urial_prompt.format(item[0]['content']) for item in df.turns]

    llm = LLM(model=model_name, revision=model_revision,trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=32, max_tokens=1024)

    outputs = llm.generate(qrys, sampling_params)
    generation = [output.outputs[0].text for output in outputs]

    df = pd.DataFrame({'instruction': qrys, 'output': generation})
    df.to_csv(f'{output_path}/{model_path}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference and save outputs to CSV.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model_revision', type=str, required=False, help='Model revision to use')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV')

    args = parser.parse_args()
    main(args.model_name, args.model_revision, args.output_path)
