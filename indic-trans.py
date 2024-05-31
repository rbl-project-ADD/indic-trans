import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransTokenizer import IndicProcessor


model_name = "ai4bharat/indictrans2-indic-indic-1B" #for indian languages translation
# model_name = "ai4bharat/indictrans2-en-indic-1B" #for english to indian languages translation
# model_name = "ai4bharat/indictrans2-indic-en-1B" #for indian languages to english translation
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

ip = IndicProcessor(inference=True)

input_sentences = [
"""अपुत्रस्य गृहं शून्यं दिशः शून्यास्त्वबांधवाः। मूर्खस्य हृदयं शून्यं सर्वशून्या दरिद्रता।।"""
]

"""
Assamese (asm_Beng)	Kashmiri (Arabic) (kas_Arab)	    Punjabi (pan_Guru)
Bengali (ben_Beng)	Kashmiri (Devanagari) (kas_Deva)	Sanskrit (san_Deva)
Bodo (brx_Deva)	    Maithili (mai_Deva)	                Santali (sat_Olck)
Dogri (doi_Deva)	Malayalam (mal_Mlym)	            Sindhi (Arabic) (snd_Arab)
English (eng_Latn)	Marathi (mar_Deva)	                Sindhi (Devanagari) (snd_Deva)
Konkani (gom_Deva)	Manipuri (Bengali) (mni_Beng)	    Tamil (tam_Taml)
Gujarati (guj_Gujr)	Manipuri (Meitei) (mni_Mtei)	    Telugu (tel_Telu)
Hindi (hin_Deva)	Nepali (npi_Deva)	                Urdu (urd_Arab)
Kannada (kan_Knda)	Odia (ory_Orya)	
"""
src_lang, tgt_lang = "san_Deva","hin_Deva"

batch = ip.preprocess_batch(
    input_sentences,
    src_lang=src_lang,
    tgt_lang=tgt_lang,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenize the sentences and generate input encodings
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations using the model
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
    )

# Decode the generated tokens into text
with tokenizer.as_target_tokenizer():
    generated_tokens = tokenizer.batch_decode(
        generated_tokens.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Postprocess the translations, including entity replacement
translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

for input_sentence, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")
