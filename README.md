pip install --upgrade gradio #to 5.34.0

#Output structure
root/
  Working_files/
    Author - BookTitle/
      TMP_AUDIO_DIR/
      TMP_M4A_DIR/
      TXT_DEBUG_DIR/
      TEMP_CONVERTED_DIR/


#Code structure
main                         # CLI launcher
└── create_gradio_app        # Initializes Gradio UI
    └── generate_btn.click() -> basic_tts   # Called when user clicks "Start"

basic_tts                    # Main pipeline to convert ebooks to audiobooks
├── detect_file_type         # Identify file type (EPUB, TXT, PDF, etc.)
├── convert_to_epub          # Convert to EPUB if needed
├── sanitize_filename        # Used multiple times to clean up filenames
├── ensure_directory         # Used multiple times for temp folders
├── extract_text_and_title_from_epub     # Extract chapters from EPUB
│   └── extract_metadata_and_cover       # Get cover image and metadata
│       └── ensure_even_dimensions       # Fix image for ffmpeg compatibility
├── russian_normalisation_accentuation   # Normalize/accent TXT input
│   └── normalize_russian                # Entry point for text normalization
│       ├── unicodedata.normalize        # Normalize text encoding (NFKC)
│       ├── process_all_caps_words       # Expand abbreviations or lower them
│       │   ├── abbreviation_exceptions  # Lookup special known abbreviations
│       │   └── pronunciation_map        # Spell out letters (e.g. С = эс)
│       ├── expand_abbreviations         # Replace common shorthand (e.g. "т.д.")
│       │   └── common_abbreviation_expansions
│       ├── normalize_dates              # Convert DD.MM.YYYY → verbal format
│       │   └── number_to_words + ordinal conversion
│       ├── currency_normalization       # Convert prices → full word form
│       │   ├── currency_to_words        # 123₽ → "сто двадцать три рубля"
│       │   └── number_to_words          # Used for rub/eur/usd/etc.
│       ├── normalize_text_with_phone_numbers # +7 (123) 456-78-90 → words
│       │   └── normalize_phone_number   # Break into segments and convert
│       ├── convert_time_expressions     # Convert 14:45 → "четырнадцать часов..."
│       ├── normalize_text_with_numbers  # Convert raw numbers → words
│       │   └── number_to_words / number_to_words_digit_by_digit
│       └── cyrrilize                    # Latin → Cyrillic transliteration
│           └── cyrrilization_mapping_extended
├── split_text_into_batches              # Split chapter text for TTS batching
├── infer                                # Run TTS on each text batch
├── embed_metadata_into_mp3              # Add metadata & cover to MP3
├── normalize_audio_folder               # Normalize MP3s before M4B (if enabled)
├── generate_chapters                    # Create chapter markers for M4B
└── convert_to_m4b                       # Combine MP3s + metadata into M4B


#add files to root F5-TTS
F5-TTS/                         # Original F5-TTS project folder
├── ru_ebook_app.py             # Main Gradio app and CLI
├── ru_ebook_files/             # Module folder
│   ├── __init__.py             # Makes this a package
│   ├── preprocessing.py        # preprocess_ru_text function
│   └── m4b_utils.py            # combine_m4a_to_m4b function



ru_ebook_app.py
ru_ebook_README.md
ru_ebook_requirements.txt
ru_ebook_libmagic
	libmagic.dll
	magic.mgc
ru_ebook_model #predownload from (F5-TTS_RUSSIANF5TTS_v1_Base)
	model_240000_inference.safetensors
	vocab.txt
ru_ebook_patch_dict.py
ru_ebook_ru4tts_normalization.py
ru_ebook_config.py

#RUN
pip install -r ru_ebook_requirements.txt

python ru_ebook_app.py --port 7860 --share


#TODO
#It might convert symbols like "№" (commonly used in Russian) to a combination of "No".
#help rewriting normalize_russian() with this token-stream-based approach?





Combine multiple batches into a larger batch
---------------------------------------
B. Use in-memory audio instead of WAV + MP3 intermediate
Right now you:

Generate .wav

Save to disk

Read with pydub

Export to .mp3

Write metadata

Delete .wav

This is I/O heavy.

Suggested:
Use io.BytesIO() for intermediate audio:

buffer = io.BytesIO()
sf.write(buffer, chapter_wave, sample_rate, format='WAV')
buffer.seek(0)
audio = AudioSegment.from_file(buffer, format="wav")
No need to hit the disk at all until final MP3.


------------------------------------------------------
 1. Avoid Repeated Code with Helper Functions
There are repeated structures like:

progress(..., desc=f"Ebook {idx+1}/{num_ebooks}: ...")
➡️ Create a utility like:

def update_progress(stage, idx, num_ebooks, offset, total, desc):
    fraction = (offset / total) + (idx / num_ebooks)
    progress(fraction, desc=f"Ebook {idx+1}/{num_ebooks}: {desc}")
	

-------------------------------------------------
🎛️ 8. Gradio UX Improvement
Show per-chapter progress bar with collapsible sections



---------------------------------
function? OR save temp files into book name folder 

        if file is None:
            logging.warning("⚠️ No file provided for preprocessing.")
            #yield gr.Textbox.update(value="⚠️ No file provided for preprocessing.")
            return "⚠️ No file provided for preprocessing."

        # Handle if 'file' is a list (multiple files)
        if isinstance(file, list):
            if len(file) == 0:
                logging.warning("⚠️ Empty file list provided.")
                #yield gr.Textbox.update(value="⚠️ Empty file list provided.")
                return "⚠️ Empty file list provided."
            file = file[0]  # Use the first file only (or change logic if you want all)
        
        original_path = file.name
        sanitized_title = sanitize_filename(os.path.splitext(os.path.basename(original_path.name))[0])