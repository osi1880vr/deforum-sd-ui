from transformers import pipeline, set_seed
import random, re


gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


def generate_prompt(starting_text):
    for count in range(4):
        seed = random.randint(100, 1000000)
        set_seed(seed)

        if starting_text == "":
            starting_text: str = line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize()
            starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)
            print(starting_text)

        response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp+'\n')

        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
        response_end = response_end.replace("<", "").replace(">", "")

        if response_end != "":
            return response_end
        if count == 4:
            return response_end

