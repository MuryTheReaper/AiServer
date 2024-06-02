import base64

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
import json
import io
from kafka import KafkaConsumer, KafkaProducer

# Load the pre-trained processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Function to process the image and extract menu info
def process_image(id, image_data):
    print("Processing image")
    image_decoded = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_decoded))
    print("Image opened")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

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

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

    result = processor.token2json(sequence)
    return extract_menu_info(id, result)


# Function to extract menu info from the result
def extract_menu_info(id, data):
    result = {'id': id}
    print(data)
    if isinstance(data, list) != True:
        newData = []
        newData.append(data)
        data = newData

    for item in data:
        try:
            if 'menu' in item:
                menu_info = []
                for menu_item in item['menu']:
                    extracted_item = {}
                    no_fields = 0
                    if 'nm' in menu_item and isinstance(menu_item['nm'], str):
                        extracted_item['name'] = menu_item['nm']
                        no_fields += 1
                    if 'unitprice' in menu_item and isinstance(menu_item['unitprice'], str):
                        try:
                            menu_item['unitprice'] = menu_item['unitprice'].replace(",", ".")
                            extracted_item['unitPrice'] = float(re.sub("[^0-9.]", "", menu_item['unitprice']))
                            no_fields += 1
                        except ValueError:
                            pass
                    if 'cnt' in menu_item and isinstance(menu_item['cnt'], str):
                        try:
                            menu_item['cnt'] = menu_item['cnt'].replace(",", ".")
                            extracted_item['count'] = float(re.sub("[^0-9.]", "", menu_item['cnt']))
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
                        if 'count' not in extracted_item:
                            extracted_item['count'] = 1.0
                        if 'unitPrice' not in extracted_item:
                            if 'price' in extracted_item:
                                extracted_item['unitPrice'] = extracted_item['price'] / extracted_item['count']
                            else:
                                extracted_item['unitPrice'] = 0.0
                        if 'price' not in extracted_item:
                            extracted_item['price'] = extracted_item['unitPrice'] * extracted_item['count']
                        menu_info.append(extracted_item)
                result['menuItems'] = menu_info
                if 'total_price' in item and isinstance(item['total_price'], str):
                    try:
                        item['total_price'] = item['total_price'].replace(",", ".")
                        total_price = float(re.sub("[^0-9.]", "", item['total_price']))
                    except ValueError:
                        total_price = 0
                    result['totalPrice'] = total_price
                elif 'total' in item and 'total_price' in item['total']:
                    try:
                        item['total']['total_price'] = item['total']['total_price'].replace(",", ".")
                        total_price = float(re.sub("[^0-9.]", "", item['total']['total_price']))
                    except ValueError:
                        total_price = 0
                    result['totalPrice'] = total_price

        except Exception as e:
            print(f"Error processing menu info: {e}")

    print(json.dumps(result, indent=2))
    return json.dumps(result)


# Kafka setup
KAFKA_BROKER_URL = '13.60.98.92:9092'
IMAGE_TOPIC = 'image-topic'
RESULT_TOPIC = 'result-topic'
GROUP_ID = 'licenta'

consumer = KafkaConsumer(
    IMAGE_TOPIC,
    bootstrap_servers=KAFKA_BROKER_URL,
    value_deserializer=lambda x: x,
    group_id=GROUP_ID,
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER_URL,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
)

for message in consumer:
    data = json.loads(message.value)
    content = data['base64Image']
    id = data['id']

    if id is None or content is None:
        print("Invalid message format, skipping...")
        continue


    print(f"Received image data with size {id}")
    result = process_image(id, content)

    producer.send(RESULT_TOPIC, value=result)
    print(f"Processed and sent result: \n")
    print(result)
