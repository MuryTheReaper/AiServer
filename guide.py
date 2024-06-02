from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
import json

# Load the pre-trained processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

# Load an image from file
image = Image.open('bon.jpg')

# Process the image
pixel_values = processor(image, return_tensors="pt").pixel_values

print(pixel_values.shape)

# Prepare the decoder input IDs with a non-empty task prompt
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

outputs = model.generate(pixel_values.to(device),
                         decoder_input_ids=decoder_input_ids.to(device),
                         max_length=model.decoder.config.max_position_embeddings,
                         early_stopping=True,
                         pad_token_id=processor.tokenizer.pad_token_id,
                         eos_token_id=processor.tokenizer.eos_token_id,
                         use_cache=True,
                         num_beams=1,
                         bad_words_ids=[[processor.tokenizer.unk_token_id]],
                         return_dict_in_generate=True,
                         output_scores=True, )

# Decode the output sequence
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

# Print the final output
result = processor.token2json(sequence)


def extract_menu_info(data):
    # Parse the JSON string
    # data = json.loads(json_str)

    # Initialize the result dictionary
    result = {}

    # Extract information from the 'menu' key
    for item in data:
        if 'menu' in item:
            menu_info = []
            for menu_item in item['menu']:
                extracted_item = {}
                no_fields = 0
                if 'nm' in menu_item and isinstance(menu_item['nm'], str):
                    extracted_item['nm'] = menu_item['nm']
                    no_fields += 1
                if 'unitprice' in menu_item and isinstance(menu_item['unitprice'], str):
                    try:
                        menu_item['unitprice'] = menu_item['unitprice'].replace(",", ".")
                        extracted_item['unitprice'] = float(re.sub("[^0-9.]", "", menu_item['unitprice']))
                        no_fields += 1
                    except ValueError:
                        pass
                if 'cnt' in menu_item and isinstance(menu_item['cnt'], str):
                    try:
                        menu_item['cnt'] = menu_item['cnt'].replace(",", ".")
                        extracted_item['cnt'] = float(re.sub("[^0-9.]", "", menu_item['cnt']))
                        no_fields += 1
                    except ValueError:
                        pass
                if 'price' in menu_item and isinstance(menu_item['price'], str):
                    try:
                        menu_item['price'] = menu_item['price'].replace(",", ".")
                        extracted_item['price'] = float(re.sub("[^0-9.]", "", menu_item['price']))
                        no_fields += 1
                    except ValueError:
                        pass
                if extracted_item and no_fields > 1:
                    if 'cnt' not in extracted_item:
                        extracted_item['cnt'] = 1.0
                    if 'unitprice' not in extracted_item:
                        if 'price' in extracted_item:
                            extracted_item['unitprice'] = extracted_item['price'] / extracted_item['cnt']
                        else:
                            extracted_item['unitprice'] = 0.0
                    if 'price' not in extracted_item:
                        extracted_item['price'] = extracted_item['unitprice'] * extracted_item['cnt']
                    menu_info.append(extracted_item)
            result['menu'] = menu_info
            # Extract the total price information
            if 'total_price' in item and isinstance(item['total_price'], str):
                try:
                    item['total_price'] = item['total_price'].replace(",", ".")
                    total_price = float(re.sub("[^0-9.]", "", item['total_price']))
                except ValueError:
                    total_price = 0
                result['total_price'] = total_price

    return json.dumps(result, indent=2)


print(extract_menu_info(result))
